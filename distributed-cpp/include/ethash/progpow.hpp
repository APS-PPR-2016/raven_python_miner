#pragma once

#include "ethash.h"
#include <cstdint>

namespace progpow {

inline constexpr int period_length = 3;
inline constexpr int num_regs = 32;
inline constexpr int num_lanes = 16;
inline constexpr int num_rounds = 64;

} // namespace progpow
