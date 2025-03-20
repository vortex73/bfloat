/**
 * @file bfloat16.hpp
 * @brief Implementation of the bfloat16 (brain float 16) data type for C++23
 * 
 * This implementation follows the format specifications of bfloat16, which preserves
 * the exponent range of float32 but reduces precision to 8 bits.
 * @author Narayan S(Vortex)
 */

#ifndef BFLOAT16_HPP
#define BFLOAT16_HPP

#include <cstdint>
#include <cmath>
#include <limits>
#include <bit>
#include <iostream>

namespace bf16 {

	class bfloat16_t {
		private:
			uint16_t data;

			// Constants for bit manipulation
			static constexpr uint32_t SIGN_MASK = 0x8000;
			static constexpr uint32_t EXP_MASK = 0x7F80;
			static constexpr uint32_t MANT_MASK = 0x007F;
			static constexpr int EXP_SHIFT = 7;
			static constexpr int EXP_BIAS = 127;

		public:
			static constexpr uint32_t get_sign_mask() { return SIGN_MASK; }
			constexpr bfloat16_t() noexcept : data(0) {}

			constexpr bfloat16_t(float value) noexcept {
				uint32_t float_bits = std::bit_cast<uint32_t>(value);

				// Round-to-nearest-even: Add 0x7FFF to the float's mantissa and then truncate
				// This adds 0.5 ULP to help with rounding
				float_bits += 0x7FFF;

				// Extract the upper 16 bits (sign, exponent, and highest 7 bits of mantissa)
				data = static_cast<uint16_t>(float_bits >> 16);
			}

			// Implicit conversion to float
			constexpr operator float() const noexcept {
				// Convert bfloat16 to float by placing its bits in the upper 16 bits
				// and setting the lower 16 bits to zero
				uint32_t float_bits = static_cast<uint32_t>(data) << 16;
				return std::bit_cast<float>(float_bits);
			}

			// Comparison operators
			auto operator<=>(const bfloat16_t& other) const noexcept = default;
			bool operator==(const bfloat16_t& other) const noexcept = default;

			// Arithmetic operators
			bfloat16_t operator-() const noexcept {
				bfloat16_t result;
				result.data = data ^ SIGN_MASK; // Flip sign bit
				return result;
			}

			bfloat16_t& operator+=(const bfloat16_t& other) noexcept {
				*this = static_cast<float>(*this) + static_cast<float>(other);
				return *this;
			}

			bfloat16_t& operator-=(const bfloat16_t& other) noexcept {
				*this = static_cast<float>(*this) - static_cast<float>(other);
				return *this;
			}

			bfloat16_t& operator*=(const bfloat16_t& other) noexcept {
				*this = static_cast<float>(*this) * static_cast<float>(other);
				return *this;
			}

			bfloat16_t& operator/=(const bfloat16_t& other) noexcept {
				*this = static_cast<float>(*this) / static_cast<float>(other);
				return *this;
			}

			// Binary arithmetic operators
			friend bfloat16_t operator+(bfloat16_t lhs, const bfloat16_t& rhs) noexcept {
				lhs += rhs;
				return lhs;
			}

			friend bfloat16_t operator-(bfloat16_t lhs, const bfloat16_t& rhs) noexcept {
				lhs -= rhs;
				return lhs;
			}

			friend bfloat16_t operator*(bfloat16_t lhs, const bfloat16_t& rhs) noexcept {
				lhs *= rhs;
				return lhs;
			}

			friend bfloat16_t operator/(bfloat16_t lhs, const bfloat16_t& rhs) noexcept {
				lhs /= rhs;
				return lhs;
			}

			// Utility functions
			bool is_nan() const noexcept {
				return ((data & EXP_MASK) == EXP_MASK) && ((data & MANT_MASK) != 0);
			}

			bool is_infinity() const noexcept {
				return ((data & EXP_MASK) == EXP_MASK) && ((data & MANT_MASK) == 0);
			}

			bool is_zero() const noexcept {
				return (data & ~SIGN_MASK) == 0;
			}

			bool is_negative() const noexcept {
				return (data & SIGN_MASK) != 0;
			}

			// Extract components
			int16_t get_exponent() const noexcept {
				if (is_zero()) return 0;
				if (is_nan() || is_infinity()) return std::numeric_limits<int16_t>::max();

				return static_cast<int16_t>(((data & EXP_MASK) >> EXP_SHIFT) - EXP_BIAS);
			}

			uint16_t get_mantissa() const noexcept {
				return data & MANT_MASK;
			}

			bool get_sign() const noexcept {
				return is_negative();
			}

			// Special value constructors
			static constexpr bfloat16_t zero() noexcept {
				bfloat16_t result;
				result.data = 0;
				return result;
			}

			static constexpr bfloat16_t infinity() noexcept {
				bfloat16_t result;
				result.data = EXP_MASK;
				return result;
			}

			static constexpr bfloat16_t negative_infinity() noexcept {
				bfloat16_t result;
				result.data = SIGN_MASK | EXP_MASK;
				return result;
			}

			static constexpr bfloat16_t nan() noexcept {
				bfloat16_t result;
				result.data = EXP_MASK | 0x0001;
				return result;
			}

			friend std::ostream& operator<<(std::ostream& os, const bfloat16_t& bf) {
				os << static_cast<float>(bf);
				return os;
			}

			// Get/set raw bits
			uint16_t bits() const noexcept {
				return data;
			}

			uint16_t& bits() noexcept {
				return data;
			}
	};

	// Math functions for bfloat16_t
	inline bfloat16_t abs(const bfloat16_t& x) noexcept {
		bfloat16_t result;
		result.bits() = x.bits() & ~bfloat16_t::get_sign_mask();
		return result;
	}

	inline bfloat16_t sqrt(const bfloat16_t& x) noexcept {
		return bfloat16_t(std::sqrt(static_cast<float>(x)));
	}

	inline bfloat16_t exp(const bfloat16_t& x) noexcept {
		return bfloat16_t(std::exp(static_cast<float>(x)));
	}

	inline bfloat16_t log(const bfloat16_t& x) noexcept {
		return bfloat16_t(std::log(static_cast<float>(x)));
	}

	inline bfloat16_t sin(const bfloat16_t& x) noexcept {
		return bfloat16_t(std::sin(static_cast<float>(x)));
	}

	inline bfloat16_t cos(const bfloat16_t& x) noexcept {
		return bfloat16_t(std::cos(static_cast<float>(x)));
	}

	inline bfloat16_t tan(const bfloat16_t& x) noexcept {
		return bfloat16_t(std::tan(static_cast<float>(x)));
	}

	inline bfloat16_t pow(const bfloat16_t& x, const bfloat16_t& y) noexcept {
		return bfloat16_t(std::pow(static_cast<float>(x), static_cast<float>(y)));
	}

	// TODO(?) SIMD(?)

} // namespace bf16

// Specialize std::numeric_limits for bfloat16_t
namespace std {
	template<>
		class numeric_limits<bf16::bfloat16_t> {
			public:
				static constexpr bool is_specialized = true;
				static constexpr bool is_signed = true;
				static constexpr bool is_integer = false;
				static constexpr bool is_exact = false;
				static constexpr bool has_infinity = true;
				static constexpr bool has_quiet_NaN = true;
				static constexpr bool has_signaling_NaN = false;
				static constexpr float_denorm_style has_denorm = denorm_present;
				static constexpr bool has_denorm_loss = true;
				static constexpr float_round_style round_style = round_to_nearest;
				static constexpr bool is_iec559 = false;
				static constexpr bool is_bounded = true;
				static constexpr bool is_modulo = false;
				static constexpr int digits = 8;
				static constexpr int digits10 = 2;
				static constexpr int max_digits10 = 4;
				static constexpr int radix = 2;
				static constexpr int min_exponent = -126;
				static constexpr int min_exponent10 = -38;
				static constexpr int max_exponent = 127;
				static constexpr int max_exponent10 = 38;
				static constexpr bool traps = false;
				static constexpr bool tinyness_before = false;

				static constexpr bf16::bfloat16_t min() noexcept { 
					bf16::bfloat16_t result;
					result.bits() = 0x0080;
					return result;
				}

				static constexpr bf16::bfloat16_t lowest() noexcept {
					bf16::bfloat16_t result;
					result.bits() = 0xFF7F;
					return result;
				}

				static constexpr bf16::bfloat16_t max() noexcept {
					bf16::bfloat16_t result;
					result.bits() = 0x7F7F;
					return result;
				}

				static constexpr bf16::bfloat16_t epsilon() noexcept {
					bf16::bfloat16_t result;
					result.bits() = 0x3C00;
					return result;
				}

				static constexpr bf16::bfloat16_t round_error() noexcept {
					return bf16::bfloat16_t(0.5f);
				}

				static constexpr bf16::bfloat16_t infinity() noexcept {
					return bf16::bfloat16_t::infinity();
				}

				static constexpr bf16::bfloat16_t quiet_NaN() noexcept {
					return bf16::bfloat16_t::nan();
				}

				static constexpr bf16::bfloat16_t denorm_min() noexcept {
					bf16::bfloat16_t result;
					result.bits() = 0x0001;
					return result;
				}
		};
}

#endif
