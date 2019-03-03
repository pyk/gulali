# Gulali
Gulali is an easy-to-use fundamental library for scientific computing with Rust,
highly inspired by [NumPy].

Documentation:
- [Quickstart Tutorial][quickstart tutorial]
- [API Reference]

[NumPy]: http://www.numpy.org/
[API Reference]: https://docs.rs/gulali

## Usage
Add this to your `Cargo.toml`:

```toml
[dependencies]
gulali = "2019.3.4"
```

and this to your crate root:

```rust
extern crate gulali;

// Import all required traits
use gulali::prelude::*;
```

To get started using Gulali, read the [quickstart tutorial].

[quickstart tutorial]:  https://docs.rs/gulali#quickstart-tutorial

## Getting help
Feel free to start discussion at [GitHub issues].

[Github issues]: https://github.com/pyk/gulali/issues/new/choose

## License
Gulali is licensed under the [Apache-2.0](./LICENSE) license.

### Contribution
Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in Gulali by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without
any additional terms or conditions.