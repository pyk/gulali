# Crabsformer
Crabsformer is an easy-to-use fundamental library for scientific computing with Rust,
highly inspired by [NumPy].

Documentation:
- [Quickstart Tutorial][quickstart tutorial]
- [API Reference]

[NumPy]: http://www.numpy.org/
[API Reference]: https://docs.rs/crabsformer

## Usage
Add this to your `Cargo.toml`:

```toml
[dependencies]
crabsformer = "2019.3.8"
```

and this to your crate root:

```rust
extern crate crabsformer;

// Import all required traits
use crabsformer::prelude::*;
```

To get started using Crabsformer, read the [quickstart tutorial].

[quickstart tutorial]:  https://docs.rs/crabsformer#quickstart-tutorial

## Getting help
Feel free to start discussion at [GitHub issues].

[Github issues]: https://github.com/pyk/crabsformer/issues/new/choose

## License
Crabsformer is licensed under the [Apache-2.0](./LICENSE) license.

### Contribution
Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in Crabsformer by you, as defined in the Apache-2.0
license, shall be licensed as above, without
any additional terms or conditions.