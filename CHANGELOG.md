# CHANGELOG

<!-- version list -->

## v1.11.0 (2025-08-09)

### Bug Fixes

- **config**: Ensure we use configuration now constants where possible
  ([`cf9584a`](https://github.com/jbus84/represent/commit/cf9584ae8453f9bc1eab4d5cf7e9f9f0618f13cd))

- **configt**: Clean is up
  ([`ca6b98f`](https://github.com/jbus84/represent/commit/ca6b98fec0f432b9009eb563e69dafc7ae9cd922))

- **constants**: Remove ununsed
  ([`1b7318a`](https://github.com/jbus84/represent/commit/1b7318a802490a0d9e2a8302ad3d6c1bbac80ee6))

- **missing-files**: Add missing files
  ([`c2b94e6`](https://github.com/jbus84/represent/commit/c2b94e607915b798683db424f064c74528d016e7))

- **normalisation**: Ensure normalisation is applied as exepcted
  ([`3e22fe0`](https://github.com/jbus84/represent/commit/3e22fe0caf09a895789a222134a9d074d6be71fb))

### Documentation

- **readme**: Updte for config
  ([`426c832`](https://github.com/jbus84/represent/commit/426c832ffd12e2260872ddfbf365bd424632adf3))

### Features

- Updated examples
  ([`2fe4129`](https://github.com/jbus84/represent/commit/2fe41299e504da91e6ac2d97ed132029b322c28f))

- **config**: Move more stuff to config
  ([`3fcbafa`](https://github.com/jbus84/represent/commit/3fcbafa997ef38d79d1fca42f15fabe765a2dbee))

- **config**: Use config everywhre
  ([`c24b410`](https://github.com/jbus84/represent/commit/c24b410b7ff8760e1bcf840091f5013a0d38dad4))

- **configs**: Fix configuration
  ([`d88eab6`](https://github.com/jbus84/represent/commit/d88eab68d791728741e73a299176d152c46338e6))

- **constants**: Jump constant rejig
  ([`26b00d2`](https://github.com/jbus84/represent/commit/26b00d21d2bf6b91438fa51b3fd17dc51b607129))

- **examples**: Commit before changing all examples
  ([`db83954`](https://github.com/jbus84/represent/commit/db83954af901e1cd3c44cc2fb790cbb2fd2e586b))

### Refactoring

- **examples**: Update for conig
  ([`0dd6deb`](https://github.com/jbus84/represent/commit/0dd6deb5634dd9fac45f34e1d3a0b37ba0e176d4))

### Testing

- Update tests
  ([`99d103f`](https://github.com/jbus84/represent/commit/99d103f62c98378fe6828b7a9eaa0de73b6f43ba))

- **examples**: Update examples
  ([`d064993`](https://github.com/jbus84/represent/commit/d064993686eeb2ee61fb439b2c1410121b59ad14))


## v1.10.0 (2025-08-05)

### Bug Fixes

- **lint**: Address ruff, mypy, and pyright errors
  ([`759b5a1`](https://github.com/jbus84/represent/commit/759b5a185035442b4896de44b645cecd39f04355))

### Documentation

- Update readme
  ([`7845702`](https://github.com/jbus84/represent/commit/78457021c1f46036e5a82f6d1f37d72d70bac6d6))

### Features

- Add dynamic classification config generation
  ([`1d39ed3`](https://github.com/jbus84/represent/commit/1d39ed3cfdaa057b6b8979306edb787c43504935))

### Refactoring

- Remove static currency config files
  ([`60cfc49`](https://github.com/jbus84/represent/commit/60cfc494d0a88a1f1c2cd801caaab8fa10a662df))


## v1.9.0 (2025-08-04)

### Features

- **parquet**: Add parquet tooling fices
  ([`f486d03`](https://github.com/jbus84/represent/commit/f486d03cc6a52ac5ee94b2fc3a15c247c46d4ab6))


## v1.8.0 (2025-08-03)

### Features

- Restore HighPerformanceDataLoader for production stability
  ([`de35f5d`](https://github.com/jbus84/represent/commit/de35f5dcb3d9d6b317cb73b0ec6827ec6996e2b7))

- **performance**: Move away from async to high perfomrance
  ([`e591652`](https://github.com/jbus84/represent/commit/e591652a2cf00ea577b87a59bd05e080d43388e7))


## v1.7.2 (2025-08-03)

### Bug Fixes

- Add version_toml configuration to ensure pyproject.toml updates
  ([`b8de258`](https://github.com/jbus84/represent/commit/b8de2589267ae121f72a644f081b4ce97c55b290))

- Update semantic-release configuration for proper __init__.py version bumping
  ([`aa11573`](https://github.com/jbus84/represent/commit/aa11573af30ac2e44df714c1c42031c80569f387))

### Build System

- **lock**: Update
  ([`0247beb`](https://github.com/jbus84/represent/commit/0247beb18fd37a1332fe19bd2d60809f231c8434))


## v1.7.1 (2025-08-02)

### Bug Fixes

- Sync __init__.py version with pyproject.toml to fix stuck version bumping
  ([`4ae7688`](https://github.com/jbus84/represent/commit/4ae7688f14a9e1e7261041212621f301e215265f))

### Continuous Integration

- **lock**: Update lock
  ([`50a8259`](https://github.com/jbus84/represent/commit/50a825908f22a04186d58f2ad7ff99a89431b674))


## v1.7.0 (2025-08-02)

### Features

- **config**: Clean up configuration logic
  ([`47376f0`](https://github.com/jbus84/represent/commit/47376f0cc5344e8de0c01d61cff9548e812286dc))


## v1.6.0 (2025-08-02)

### Features

- Implement working Pydantic configuration system
  ([`cee83e0`](https://github.com/jbus84/represent/commit/cee83e0bf33acc9d1b45081eb9370c16ce8e3790))

- **configs**: Add configurations
  ([`f70078a`](https://github.com/jbus84/represent/commit/f70078a20c5eafb202e57097b43d133ec8ac167b))


## v1.5.0 (2025-08-02)

### Features

- **classification**: Add classification tooling
  ([`e8da149`](https://github.com/jbus84/represent/commit/e8da149de52751e6c6bdbaeeb3582870da56705b))

- **targets**: Make sure targets are produced by the dataloader
  ([`012d37f`](https://github.com/jbus84/represent/commit/012d37f0396e6867f9521165332c97899830f000))


## v1.4.0 (2025-07-30)

### Features

- Comprehensive README update with uv installation and PyTorch examples
  ([`20e1bed`](https://github.com/jbus84/represent/commit/20e1bedfb65ef20c05cbbf42c09fcd28eea5e9d2))


## v1.3.0 (2025-07-30)

### Features

- **features**: Add new target features
  ([`af85bd1`](https://github.com/jbus84/represent/commit/af85bd154c76a237719e80cae830e5e0394e8fcd))


## v1.2.0 (2025-07-29)

### Documentation

- **readme**: Update the readme
  ([`5035a7b`](https://github.com/jbus84/represent/commit/5035a7b9f15ee114120528ef9eefcfe0b14cf1b7))

### Features

- **dataloader**: Add dataloader and tests
  ([`cb4b375`](https://github.com/jbus84/represent/commit/cb4b375a4a03ad64c079f38a619729d14767ab50))

- **example**: Add pytorch dataloading examples
  ([`34a6df4`](https://github.com/jbus84/represent/commit/34a6df45f26bfe0a194c4f01ea790ea664907e92))


## v1.1.0 (2025-07-29)

### Features

- **examples**: Add new exaqmples output
  ([`43032c4`](https://github.com/jbus84/represent/commit/43032c4d80b7ab5aa9381dbd5d29a71c9d58e36e))

- **viz**: Add visualisation example
  ([`a2efada`](https://github.com/jbus84/represent/commit/a2efadaff16aed84c5503de8ead29a5a18109fe8))

### Testing

- Gemini fix up of tests
  ([`fe12b24`](https://github.com/jbus84/represent/commit/fe12b2418c53e3880a4a4505c8da0c8fe6092be3))


## v1.0.1 (2025-07-29)

### Bug Fixes

- Add missing make target
  ([`e04c2a4`](https://github.com/jbus84/represent/commit/e04c2a4de67ca86d2fdb42808fd454a4916adb9f))

- Add pre-commit
  ([`057a617`](https://github.com/jbus84/represent/commit/057a61725b15405a464efc1220616b68226855e3))


## v1.0.0 (2025-07-29)

- Initial Release
