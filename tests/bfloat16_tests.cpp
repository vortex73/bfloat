/**
 * @file bfloat16_tests.cpp
 * @brief Test suite for bfloat16_t implementation using Catch2
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <bfloat16/bfloat16.hpp>
#include <array>
#include <cmath>
#include <limits>

using namespace bf16;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

auto float_matcher(float expected) {
	if (std::isnan(expected)) {
		return WithinAbs(0.0f, 1.0f);
	} else if (std::isinf(expected)) {
		return WithinAbs(expected, 0.1f);
	} else if (std::abs(expected) < 1e-5f) {
		return WithinAbs(expected, 1e-3f);
	} else {
		return WithinRel(expected, 0.01f);
	}
}

bool is_nan(float value) {
	return std::isnan(value);
}

TEST_CASE("BFloat16 Conversions", "[bfloat16][conversion]") {
	SECTION("Basic conversion") {
		std::array<float, 14> test_values = {
			0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 
			3.14159f, -3.14159f, 
			1e-6f, -1e-6f,
			1e6f, -1e6f,
			std::numeric_limits<float>::infinity(), 
			-std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::quiet_NaN()
		};

		for (float f : test_values) {
			bfloat16_t bf = f;
			float back_to_float = static_cast<float>(bf);

			// Special handling for NaN comparison
			if (std::isnan(f)) {
				REQUIRE(std::isnan(back_to_float));
			} else {
				REQUIRE_THAT(back_to_float, float_matcher(f));
			}
		}
	}

	SECTION("Round-trip identity for common values") {
		// These values should convert exactly in bfloat16
		std::array<float, 5> exact_values = {
			0.0f, 1.0f, 2.0f, 4.0f, 8.0f
		};

		for (float f : exact_values) {
			bfloat16_t bf = f;
			float back_to_float = static_cast<float>(bf);
			REQUIRE(back_to_float == f);
		}
	}
}

TEST_CASE("BFloat16 Arithmetic", "[bfloat16][arithmetic]") {
	SECTION("Addition") {
		bfloat16_t a(3.5f);
		bfloat16_t b(1.5f);
		bfloat16_t sum = a + b;
		REQUIRE_THAT(static_cast<float>(sum), float_matcher(5.0f));
	}

	SECTION("Subtraction") {
		bfloat16_t a(3.5f);
		bfloat16_t b(1.5f);
		bfloat16_t diff = a - b;
		REQUIRE_THAT(static_cast<float>(diff), float_matcher(2.0f));
	}

	SECTION("Multiplication") {
		bfloat16_t a(3.5f);
		bfloat16_t b(2.0f);
		bfloat16_t prod = a * b;
		REQUIRE_THAT(static_cast<float>(prod), float_matcher(7.0f));
	}

	SECTION("Division") {
		bfloat16_t a(3.5f);
		bfloat16_t b(2.0f);
		bfloat16_t quot = a / b;
		REQUIRE_THAT(static_cast<float>(quot), float_matcher(1.75f));
	}

	SECTION("Compound Assignment") {
		bfloat16_t a(10.0f);
		bfloat16_t b(3.5f);

		a += b;
		REQUIRE_THAT(static_cast<float>(a), float_matcher(13.5f));

		a -= b;
		REQUIRE_THAT(static_cast<float>(a), float_matcher(10.0f));

		a *= b;
		REQUIRE_THAT(static_cast<float>(a), float_matcher(35.0f));

		a /= b;
		REQUIRE_THAT(static_cast<float>(a), float_matcher(10.0f));
	}

	SECTION("Negation") {
		bfloat16_t a(3.5f);
		bfloat16_t neg_a = -a;

		REQUIRE_THAT(static_cast<float>(neg_a), float_matcher(-3.5f));
	}
}

TEST_CASE("BFloat16 Math Functions", "[bfloat16][math]") {
	SECTION("Absolute Value") {
		bfloat16_t a(-3.5f);
		bfloat16_t abs_a = abs(a);

		REQUIRE_THAT(static_cast<float>(abs_a), float_matcher(3.5f));
	}

	SECTION("Square Root") {
		bfloat16_t a(16.0f);
		bfloat16_t sqrt_a = sqrt(a);

		REQUIRE_THAT(static_cast<float>(sqrt_a), float_matcher(4.0f));
	}

	SECTION("Trigonometric Functions") {
		bfloat16_t a(0.0f);

		REQUIRE_THAT(static_cast<float>(sin(a)), float_matcher(0.0f));
		REQUIRE_THAT(static_cast<float>(cos(a)), float_matcher(1.0f));
		REQUIRE_THAT(static_cast<float>(tan(a)), float_matcher(0.0f));
	}

	SECTION("Exponential and Logarithm") {
		bfloat16_t a(1.0f);

		REQUIRE_THAT(static_cast<float>(exp(a)), float_matcher(std::exp(1.0f)));
		REQUIRE_THAT(static_cast<float>(log(a)), float_matcher(0.0f));
	}

	SECTION("Power Function") {
		bfloat16_t a(2.0f);
		bfloat16_t b(3.0f);

		REQUIRE_THAT(static_cast<float>(pow(a, b)), float_matcher(8.0f));
	}
}

TEST_CASE("BFloat16 Special Values", "[bfloat16][special]") {
	SECTION("Zero") {
		bfloat16_t zero = bfloat16_t::zero();
		REQUIRE(zero.is_zero());
		REQUIRE_FALSE(zero.is_negative());
	}

	SECTION("Negative Zero") {
		bfloat16_t neg_zero(-0.0f);
		REQUIRE(neg_zero.is_zero());
		REQUIRE(neg_zero.is_negative());
	}

	SECTION("Infinity") {
		bfloat16_t inf = bfloat16_t::infinity();
		REQUIRE(inf.is_infinity());
		REQUIRE_FALSE(inf.is_negative());

		bfloat16_t neg_inf = bfloat16_t::negative_infinity();
		REQUIRE(neg_inf.is_infinity());
		REQUIRE(neg_inf.is_negative());
	}

	SECTION("NaN") {
		bfloat16_t nan_val = bfloat16_t::nan();
		REQUIRE(nan_val.is_nan());
	}

	SECTION("Operations with Special Values") {
		bfloat16_t normal(1.0f);
		bfloat16_t inf = bfloat16_t::infinity();
		bfloat16_t nan_val = bfloat16_t::nan();

		// Normal + Infinity = Infinity
		REQUIRE((normal + inf).is_infinity());

		// Infinity + Infinity = Infinity
		REQUIRE((inf + inf).is_infinity());

		// Normal * Infinity = Infinity
		REQUIRE((normal * inf).is_infinity());

		// Operations with NaN result in NaN
		REQUIRE((normal + nan_val).is_nan());
		REQUIRE((inf + nan_val).is_nan());
	}
}

TEST_CASE("BFloat16 Numeric Limits", "[bfloat16][limits]") {
	SECTION("Min and Max Values") {
		auto min_val = std::numeric_limits<bfloat16_t>::min();
		auto max_val = std::numeric_limits<bfloat16_t>::max();

		REQUIRE(static_cast<float>(min_val) > 0.0f);
		REQUIRE(static_cast<float>(max_val) > 0.0f);
		REQUIRE(static_cast<float>(max_val) > static_cast<float>(min_val));
	}

	SECTION("Special Values from Limits") {
		auto inf = std::numeric_limits<bfloat16_t>::infinity();
		auto nan_val = std::numeric_limits<bfloat16_t>::quiet_NaN();

		REQUIRE(inf.is_infinity());
		REQUIRE(nan_val.is_nan());
	}

	SECTION("Epsilon") {
		auto epsilon = std::numeric_limits<bfloat16_t>::epsilon();

		// Epsilon should be positive and small
		REQUIRE(static_cast<float>(epsilon) > 0.0f);

		// 1.0 + epsilon should be different from 1.0
		REQUIRE(static_cast<float>(bfloat16_t(1.0f) + epsilon) > 1.0f);
	}
}

TEST_CASE("BFloat16 Bit Patterns", "[bfloat16][bits]") {
	SECTION("Bit Layout for Common Values") {
		// Zero: All bits are 0
		REQUIRE(bfloat16_t(0.0f).bits() == 0x0000);

		// One: Exponent is 127 (bias), mantissa is 0
		REQUIRE(bfloat16_t(1.0f).bits() == 0x3F80);

		// Negative One: Sign bit set, exponent is 127, mantissa is 0
		REQUIRE(bfloat16_t(-1.0f).bits() == 0xBF80);

		// Two: Exponent is 128 (bias + 1), mantissa is 0
		REQUIRE(bfloat16_t(2.0f).bits() == 0x4000);

		// Infinity: Exponent all 1s, mantissa 0
		REQUIRE(bfloat16_t::infinity().bits() == 0x7F80);
	}

	SECTION("Component Access") {
		bfloat16_t a(1.5f);  // 1.5 = 1 + 0.5 = 1 + 2^-1

		// For 1.5, sign is 0, exponent is 127 (bias), mantissa has 2nd bit set
		REQUIRE_FALSE(a.is_negative());
		REQUIRE(a.get_exponent() == 0);  // 2^0
		REQUIRE(a.get_mantissa() == 0x40);  // 0.5 in the mantissa (first bit after implied 1)
	}
}

TEST_CASE("BFloat16 Precision Loss", "[bfloat16][precision]") {
	SECTION("Precision Loss Demonstration") {
		// These numbers are close but should be distinct in float32
		float f1 = 1.0f;
		float f2 = 1.0f + 1e-7f;

		// After conversion to bfloat16, they should be the same due to precision loss
		bfloat16_t bf1(f1);
		bfloat16_t bf2(f2);

		REQUIRE(bf1.bits() == bf2.bits());

		// Converting back should give identical values
		float back1 = static_cast<float>(bf1);
		float back2 = static_cast<float>(bf2);

		REQUIRE(back1 == back2);
	}

	SECTION("Range Preservation") {
		// BFloat16 should preserve large values that float16 would overflow
		float large_value = 1.0e20f;

		bfloat16_t bf_large(large_value);

		// The converted value should be finite and have the same order of magnitude
		REQUIRE_FALSE(bf_large.is_infinity());

		float back = static_cast<float>(bf_large);

		// Ordering should be preserved for large values
		REQUIRE((back > 0.0f && large_value > 0.0f) || (back < 0.0f && large_value < 0.0f));

		// Order of magnitude should be close
		double ratio = std::abs(back / large_value);
		REQUIRE(ratio > 0.5);
		REQUIRE(ratio < 2.0);
	}
}
