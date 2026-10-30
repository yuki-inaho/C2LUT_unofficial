// SPDX-License-Identifier: MIT
#include "trilinear.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;

namespace {

using Input = nb::ndarray<nb::numpy, const double, nb::c_contig, nb::device::cpu>;
using Output = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;

void check_alignment(const Input& array) {
    if (array.size() != 0 && reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0) {
        throw std::invalid_argument("arrays must be aligned for float64 values");
    }
}

Output interpolate(const Input& lut, const Input& points) {
    if (lut.ndim() != 4 || lut.shape(0) != 3 || lut.shape(1) != lut.shape(2)
        || lut.shape(1) != lut.shape(3)) {
        throw std::invalid_argument("lut must have shape (3, N, N, N)");
    }
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (M, 3)");
    }
    check_alignment(lut);
    check_alignment(points);
    const std::size_t count = points.shape(0);
    const std::size_t output_size = c2lut::checked_product(count, 3);
    // Allocate one element for empty outputs so the ownership handle is valid.
    auto data = std::make_unique<double[]>(std::max(output_size, std::size_t{1}));
    {
        nb::gil_scoped_release release;
        c2lut::trilinear(lut.data(), lut.shape(1), points.data(), count, data.get());
    }
    // The capsule owns the allocation, independently of both input arrays.
    nb::capsule owner(data.get(), [](void* pointer) noexcept {
        delete[] static_cast<double*>(pointer);
    });
    auto* pointer = data.release();
    return Output(pointer, {count, 3}, owner);
}

}  // namespace

NB_MODULE(_native, module) {
    module.doc() = "Portable C++17 trilinear interpolation for C2LUT.";
    module.def("trilinear", &interpolate, nb::arg("lut").noconvert(),
               nb::arg("points").noconvert(),
               "Interpolate float64 C-contiguous LUT (3,N,N,N) at points (M,3).\n\n"
               "Points must be finite and within [0,1]; all LUT entries must be finite.\n"
               "The result owns its data. Computation releases the Python GIL.\n"
               "Raises ValueError for invalid shapes/values and TypeError for incompatible buffers.");
}
