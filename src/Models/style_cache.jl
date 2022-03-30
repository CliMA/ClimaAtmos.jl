
"""
Supertype for all model cache styles.
"""
abstract type AbstractCacheStyle <: AbstractModelStyle end

struct CacheEmpty <: AbstractCacheStyle end
struct CacheBase <: AbstractCacheStyle end
struct CacheZeroMomentMicro <: AbstractCacheStyle end
