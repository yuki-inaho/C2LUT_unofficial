// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace c2lut {

// The core deliberately has no dependency on Python, NumPy, or a thread runtime.
inline std::size_t checked_product(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument("array dimensions overflow size_t");
    }
    return left * right;
}

struct CellCoordinate {
    std::size_t lower;
    double fraction;
};

inline CellCoordinate locate(double value, std::size_t resolution) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument("points must be finite and within [0, 1]");
    }
    const double position = value * static_cast<double>(resolution - 1);
    // At value == 1, use the final cell and its upper vertex. Never read N.
    const auto lower = std::min(static_cast<std::size_t>(position), resolution - 2);
    return {lower, position - static_cast<double>(lower)};
}

/**
 * Interpolate a channel-first LUT of shape (3, N, N, N) at M RGB points.
 *
 * All buffers contain double values in row-major order. The output has shape
 * (M, 3), must not alias the inputs, and is supplied by the caller. The LUT is
 * validated even when M == 0. Input/output buffers may be null only for M == 0.
 * Invalid dimensions and non-finite/out-of-domain data throw invalid_argument.
 */
inline void trilinear(const double* lut, std::size_t resolution,
                      const double* points, std::size_t point_count,
                      double* output) {
    if (resolution < 2) {
        throw std::invalid_argument("LUT resolution must be at least 2");
    }
    if (lut == nullptr || (point_count != 0 && (points == nullptr || output == nullptr))) {
        throw std::invalid_argument("non-empty arrays require valid buffers");
    }
    const std::size_t plane = checked_product(checked_product(resolution, resolution), resolution);
    const std::size_t lut_size = checked_product(3, plane);
    checked_product(point_count, 3);
    for (std::size_t index = 0; index < lut_size; ++index) {
        if (!std::isfinite(lut[index])) {
            throw std::invalid_argument("LUT values must be finite");
        }
    }

    for (std::size_t pixel = 0; pixel < point_count; ++pixel) {
        const std::array<CellCoordinate, 3> cell = {
            locate(points[3 * pixel], resolution),
            locate(points[3 * pixel + 1], resolution),
            locate(points[3 * pixel + 2], resolution),
        };
        const std::array<std::array<double, 2>, 3> weights = {{
            {1.0 - cell[0].fraction, cell[0].fraction},
            {1.0 - cell[1].fraction, cell[1].fraction},
            {1.0 - cell[2].fraction, cell[2].fraction},
        }};
        std::array<double, 3> value{};
        for (std::size_t red = 0; red < 2; ++red) {
            for (std::size_t green = 0; green < 2; ++green) {
                for (std::size_t blue = 0; blue < 2; ++blue) {
                    const double weight = weights[0][red] * weights[1][green] * weights[2][blue];
                    const std::size_t offset =
                        ((cell[0].lower + red) * resolution + cell[1].lower + green) * resolution
                        + cell[2].lower + blue;
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        value[channel] += weight * lut[channel * plane + offset];
                    }
                }
            }
        }
        for (std::size_t channel = 0; channel < 3; ++channel) {
            output[3 * pixel + channel] = value[channel];
        }
    }
}

}  // namespace c2lut
