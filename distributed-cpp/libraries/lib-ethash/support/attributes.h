/* ethash: C/C++ implementation of Ethash, the Ethereum Proof of Work algorithm.
 * Copyright 2018-2019 Pawel Bylica.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

/** inline */
#if _MSC_VER || __STDC_VERSION__
#define INLINE inline
#else
#define INLINE
#endif

/** [[always_inline]] */
#if _MSC_VER
#define ALWAYS_INLINE __forceinline
#elif defined(__has_attribute) && __STDC_VERSION__
#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((always_inline))
#endif
#endif
#if !defined(ALWAYS_INLINE)
#define ALWAYS_INLINE
#endif

/** [[no_sanitize()]] */
#if __clang__
#define NO_SANITIZE(sanitizer) \
    __attribute__((no_sanitize(sanitizer)))
#else
#define NO_SANITIZE(sanitizer)
#endif

/** NOEXCEPT */
#ifndef NOEXCEPT
#ifdef __cplusplus
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif
#endif

#ifndef ATTRIBUTE_NOINLINE
#if _MSC_VER
#define ATTRIBUTE_NOINLINE __declspec(noinline)
#elif defined(__has_attribute)
#if __has_attribute(noinline)
#define ATTRIBUTE_NOINLINE __attribute__((noinline))
#endif
#endif
#if !defined(ATTRIBUTE_NOINLINE)
#define ATTRIBUTE_NOINLINE
#endif
#endif
