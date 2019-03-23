<p align="center">
  <img alt="Crabsformer. Larva Island S01E06" src="crabsformer.png">
  <p align="center" style="font-size:32px;"><b>Crabsformer</b></p>
  <p align="center">
    <i>
    Crabsformer is an easy-to-use fundamental library for
    scientific computing with Rust, highly inspired by
    <a href="http://www.numpy.org/">NumPy</a>.
    </i>
  </p>
</p>

**Notice!** This project is in early phase. Expect bugs and missing features.

## Documentation
- [Quickstart Tutorial][quickstart tutorial]
- [API Reference]

[NumPy]: http://www.numpy.org/
[API Reference]: https://docs.rs/crabsformer

## Usage
Add this to your `Cargo.toml`:

```toml
[dependencies]
crabsformer = "2019.3.13"
```

and this to your crate root:

```rust
#[macro_use] extern crate crabsformer;
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
