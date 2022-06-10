def _get_password_hasher():
    try:
        import argon2

        # Parameters are picked based on the tests run on an t2.micro with a single core, 1GB of RAM & 1GB of swap
        # Parameters were picked to have a password hashing time less than 0.5seconds, while utilizing 32MB of memory
        return argon2.PasswordHasher(time_cost=14, memory_cost=32768, parallelism=2, hash_len=64)
    except ImportError:
        return None


PASSWORD_HASHER = _get_password_hasher()


def create_hash(password: str) -> str:
    return PASSWORD_HASHER.hash(password)


def verify_password(password: str, hashed_password: str) -> (bool, bool):
    """Verify password hash

    :returns is the password valid & whether the password should be rehashed based on the new hashing parameters
    """
    try:
        valid = PASSWORD_HASHER.verify(hashed_password, password)
        should_rehash = PASSWORD_HASHER.check_needs_rehash(hashed_password)

        return valid, should_rehash
    except Exception:
        return False, False
