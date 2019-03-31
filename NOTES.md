# Data Type?
Do we need introduce new data type for multidimensional array?
I think we don't need that.

Let's just use:

    Vec<Vec<T>> as 2D Array
    Vec<Vec<Vec<T>>> as 3D Array

Our focus will be on 2D array and 3D array creation like:

    zeros()

that can return 2D array filled with zeros. Is it possible?
we will se.

If we only use `Vec<Vec<T>>` or `Vec<Vec<Vec<T>>>`, we will
not have an information about the array it self. Like, it is
two dimensional or tree dimensional?

Yeah, but we don't need to keep that information right?
let's see.

btw, we can't use one function `zeros()` to build zero-valued
two dimensional and three dimensional array because rust
is [doesn't support function overloading](https://blog.rust-lang.org/2015/05/11/traits.html).

> *Overloading*. Rust does not support traditional overloading
> where the same method is defined with multiple signatures.
> But traits provide much of the benefit of overloading: if
> a method is defined generically over a trait, it can be
> called with any type implementing that trait. Compared to
> traditional overloading, this has two advantages. First,
> it means the overloading is less ad hoc: once you understand
> a trait, you immediately understand the overloading pattern
> of any APIs using it. Second, it is extensible: you can
> effectively provide new overloads downstream from a method
> by providing new trait implementations.

So, we can use trait to solve this right? but how?

# with_size, with_shape, one_dim, two_dim ...

We will replace `Vec::with_size(x)` and `Vec::with_shape(&[x])`
with `Vec::one_dim(x)`, `Vec::two_dim(a,b)`, `Vec::three_dim(a,b,c)`
and `Vec::four_dim(a,b,c,d)`. I think it's more intuitive.

For example:

```
let a: Vec<Vec<i64>> = Vec::two_dim(x, y)
```

It clearly says, initialize 2D vector. Oke, Nice!

# Developer Experience first
I think we already got the basics, we need to follow numpy docs
first before continue adding new feature.

Read the Numpy docs -> Modify the docs for `np` -> Write
and example.

# Store N-Dimensional Array as One array
Btw, we can store n-dimensional array as one array.

> More generally, in a k-dimensional array, the address of an
> element with indices i1, i2, ..., ik is
>
> B + c1 · i1 + c2 · i2 + ... + ck · ik.
>

From [wikipedia](https://en.wikipedia.org/wiki/Array_data_structure#Multidimensional_arrays)

If we build something like that, we can access the data using `arr[i][j][k]`
syntax right? I don't know let's try.

We stuck on this

    impl<T> Index<usize> for Array<T> {
        type Output = T;

        fn index(&self, index: usize) -> &T {
            &self.data[index]
            // Index::index(&*self, index)
        }
    }

How to implement the `[i][j]` part? if we returns the element
then we can't do reindex. What we need is:

    dimension -> syntax -> expected returns type
    if ndim == 1 -> arr[i] -> T
    if ndim == 2 -> arr[i] -> Arrray<T>
    if ndim == 2 -> arr[i][j] -> T
    if ndim == 3 -> arr[i] -> Arrray<T>
    if ndim == 3 -> arr[i][j] -> Arrray<T>
    if ndim == 3 -> arr[i][j][k] -> T

Btw berarti kalo mau `arr[i][j]` berarti `Array<Array<T>>` dong ya?
bener juga hmmm ...

nah yg tak pengen dia tetep `Array<T>` si.


# Rust Module
Btw, we can move the vector builder to it's own module: `builder`.

Example of code the structure: [rand]

[rand]: https://github.com/rust-random/rand/tree/master/src

and we can split the builder into their own file like: `zeros.rs`,
`ones.rs`, `full.rs` and so on. We will add the test on each file
like the following.

# Rename the project to Gibbs

Gibbs is named after [Josiah Willard Gibbs](https://www.britannica.com/science/vector-mathematics).

Hmmmmmmm

The name is very confusing:
[https://www.google.com/search?q=Gibbs](https://www.google.com/search?q=Gibbs).

I think it's not appropriate to use that name. Let's brainstorming first:

```
vector
multidimensional
vector extension
VectorX
```

Anw we can use [name generator](https://namelix.com/app/?keywords=science+vector+operations)
wkwk.

```
vectwise
hivector
solvect
savec
wavec
covect
solvecto
covecto
eigenic
```

fuk ternyata susah banget cari nama ASW.
udah ah np aja. tai.

I think I'm gonna use Gulali as a name,
Gulali is an Indonesian-flavored candy.
It's sweet and everyone favorite. I think it
describe how this library so swell. I want to
make library that easy-to-use and learn.


# Vector creation

We will use [Numpy Array Creation Routines] as the reference.

We already implement:

```
ones()
zeros()
full()
```

NumPy array creation routines:

```
empty(shape[, dtype, order])	Return a new array of given shape and type, without initializing entries.
empty_like(prototype[, dtype, order, subok])	Return a new array with the same shape and type as a given array.
eye(N[, M, k, dtype, order])	Return a 2-D array with ones on the diagonal and zeros elsewhere.
identity(n[, dtype])	Return the identity array.
ones_like(a[, dtype, order, subok])	Return an array of ones with the same shape and type as a given array.
zeros_like(a[, dtype, order, subok])	Return an array of zeros with the same shape and type as a given array.
full_like(a, fill_value[, dtype, order, subok])	Return a full array with the same shape and type as a given array.
```

anw kita fokus ke [Numpy Array Creation] dulu untuk edit `src/builders/mod.rs`.
keyh.


[Numpy Array Creation]: https://docs.scipy.org/doc/numpy-1.16.1/user/basics.creation.html
[Numpy Array Creation Routines]: https://docs.scipy.org/doc/numpy-1.16.1/reference/routines.array-creation.html

Let's create `np.arange` API:

```
numpy.arange([start, ]stop, [step, ]dtype=None) -> returns 1D array
```

```
>>> np.arange(3)
array([0, 1, 2])
>>> np.arange(3.0)
array([ 0.,  1.,  2.])
>>> np.arange(3,7)
array([3, 4, 5, 6])
>>> np.arange(3,7,2)
array([3, 5])
```

So maybe we ...


Oke `range()` udah. Sip.

# Init vs Generate
Currently we use the following API to create range vector:

```rust
let range2: Vec<f64> = Vec::range()
    .start_at(1.0)
    .stop_at(3.0)
    .init();
assert_eq!(range2, [1.0, 2.0]);
```

Inspired by [MATLAB linspace()], we can use `generate` instead
of `init`. It is far more intuitive. Ok.

[MATLAB linspace()]: https://www.mathworks.com/help/matlab/ref/linspace.html


# Rust error handling
btw how to handle the error?

I have a function like the following:

```
pub fn generate(self) -> Vec<T>
```

it generates `Vec<T>` based on `self`.

There are two cases:
1. `stop` value should be specified. otherwise it returns None.
2. `start >= stop` it returns None.

# API Dilemma
So, currently we have the following API to generate multidimensional
vector:

```rust
let arr: Vec<u8> = Vec::one_dim(2).ones();
let arr: Vec<Vec<u8>> = Vec::two_dim(2, 2).ones();
let arr: Vec<Vec<Vec<u8>>> = Vec::three_dim(2, 2, 2).ones();
let arr: Vec<Vec<Vec<Vec<u8>>>> = Vec::four_dim(2, 2, 2, 2).ones();
```

and the following api to generate numerical range vector:

```rust
let range: Vec<f64> = Vec::range()
    .start_at(1.0)
    .stop_at(3.0)
    .init();

let lin: Vec<f64> = Vec::linspace()
    .start_at(1.0)
    .stop_at(3.0)
    .with_size(4)
    .generate()
    .unwrap();
```

first of all it's not consistent.

Currently:

```rust
// Create two-dimensional vector with shape [3, 3]
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::two_dim(3, 3).zeros();
```

We should say **Generate** for consistency:

```rust
// Generate n-dimensional vector with shape [3, 3]
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::zeros()
    .with_shape([3, 3])
    .generate();
```

anw how about `full`?

```rust
let matrix: Vec<Vec<i32>> = Vec::full()
    .of_value(2.5)
    .with_shape([2, 2])
    .generate();
```

hmmm i think it's not intuitive. Let's get inspiration.

Numpy

```python
# Return a new array of given shape and type, filled with fill_value.
np.full((2, 2), np.inf)
np.full((2, 2), 10)
```

hmm I can't find any
[related function](https://www.mathworks.com/help/matlab/matrices-and-arrays.html)
in matlab.

Wording:

- Return a new array of given shape and type, filled with `fill_value`.
- Generate a new vector of given shape, filled with `fill_value`.

```rust
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new m-dimensional vector of given shape,
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .zeros()
    .generate();
```

Mantul! Jadi gini ya

```rust
// Generate a new n-dimensional vector of given shape,
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .zeros()
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with ones
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .ones()
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with 2
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with 2
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new range vector
let range: Vec<f64> = Vec::new()
    .range()
    .start_at(1.0)
    .stop_at(3.0)
    .generate();

// Generate a new linearly spaced vector
let lin: Vec<f64> = Vec::new()
    .linspace()
    .start_at(1.0)
    .stop_at(3.0)
    .with_size(4)
    .generate();
```

Btw, we can't use `Vec::new()` because it returns
`Vec<T>`. Not works for n-dimensional vector.

How about:

```rust
Vec::ndim(); // n-dimensional vector
Vec::new_ndim() // new n-dimensional vector;
```

So, we will revisit the API like the following:


```rust
// Generate a new n-dimensional vector of given shape,
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::ndim()
    .with_shape([2, 2])
    .zeros()
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with ones
let matrix: Vec<Vec<i32>> = Vec::ndim()
    .with_shape([2, 2])
    .ones()
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with 2
let matrix: Vec<Vec<i32>> = Vec::ndim()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with 2
let matrix: Vec<Vec<i32>> = Vec::ndim()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new range vector
let range: Vec<f64> = Vec::ndim()
    .range()
    .start_at(1.0)
    .stop_at(3.0)
    .generate();

// Generate a new linearly spaced vector
let lin: Vec<f64> = Vec::ndim()
    .linspace()
    .start_at(1.0)
    .stop_at(3.0)
    .with_size(4)
    .generate();
```

Oh forgot, we can't use the same method trait like this:

```
let arr: Vec<u8> = Vec::ndim();
let arr: Vec<Vec<u8>> = Vec::ndim();
let arr: Vec<Vec<Vec<u8>>> = Vec::ndim();
let arr: Vec<Vec<Vec<Vec<u8>>>> = Vec::ndim();
```

if we implement `ndim` for other `Vec<T>`, it will raise
a conflict:

```
error[E0119]: conflicting implementations of trait `builders::NDimensional` for type `std::vec::Vec<std::vec::Vec<_>>`:
   --> src/builders/mod.rs:167:1
    |
158 | / impl<T> NDimensional for Vec<T>
159 | | where
160 | |     T: Num + FromPrimitive + Copy,
161 | | {
...   |
164 | |     }
165 | | }
    | |_- first implementation here
166 |
167 | / impl<T> NDimensional for Vec<Vec<T>>
168 | | where
169 | |     T: Num + FromPrimitive + Copy,
170 | | {
...   |
173 | |     }
174 | | }
    | |_^ conflicting implementation for `std::vec::Vec<std::vec::Vec<_>>`
    |
    = note: upstream crates may add new impl of trait `num::Num` for type `std::vec::Vec<_>` in future versions
    = note: upstream crates may add new impl of trait `num::FromPrimitive` for type `std::vec::Vec<_>` in future versions
    = note: upstream crates may add new impl of trait `std::marker::Copy` for type `std::vec::Vec<_>` in future versions
```

so our previous solution works perfectly:

```
let arr: Vec<u8> = Vec::one_dim();
let arr: Vec<Vec<u8>> = Vec::two_dim();
let arr: Vec<Vec<Vec<u8>>> = Vec::three_dim();
let arr: Vec<Vec<Vec<Vec<u8>>>> = Vec::four_dim();
```

OK.

Hmmm let's modify it a litle bit.

Instead of returns `Vec<T>` in `one_dim()`,
we may return a new struct.

# Rename it again to Crabsformer
Inspired by [Larva Island S01E06: Crabsformer](https://www.imdb.com/title/tt9152116/).

Why?

1. [The rust most used mascot is a crab](http://rustacean.net/)
2. If you are stranded in an island, you can use Crabsformer to
   build a ship. See [Larva Island S01E06](https://www.imdb.com/title/tt9152116/).

Also, it's a very good name. wkwk.
Keyh.

let's rename it.
We need to update the repo and update the `gulali` docs. Keyh.

# Distribution type
So I want to be able todo the following:

```rust
let a: Vec<f32> = Vec::one_dim()
    .with_shape([5])
    .random()
    .with_distribution(dist)
    .generate();
```

Pokoknya harus bisa kaya gini si:

```rust
use rand::distributions::{Distribution, Uniform};

fn main() {
    let between = Uniform::from(10..10000);
    let mut rng = rand::thread_rng();
    let mut sum = 0;
    for _ in 0..1000 {
        sum += between.sample(&mut rng);
    }
    println!("{}", sum);
}
```

Harusnya kaya gini bisa bosku

```rust
#[derive(Debug)]
struct Test1<T> {
    data: T
}

#[derive(Debug)]
struct Test2<T> {
    data: T
}

trait TraitTest<T> {
    fn set(&self, new_data: T) -> T;
}

impl<T> TraitTest<T> for Test1<T> {
    fn set(&self, new_data: T) -> T {
        new_data
    }
}

impl<T> TraitTest<T> for Test2<T> {
    fn set(&self, new_data: T) -> T {
        new_data
    }
}

#[derive(Debug)]
struct ContainerStruct<T, D>
    where D: TraitTest<T>
{
    data: T,
    trait_ok: Option<D>
}

fn main() {
    let t1 = Test1{data: 1};
    println!("{:?}", t1);
    println!("{:?}", t1.set(12));

    // Kita coba container struct
    let c1 = ContainerStruct{
        data: 12,
        trait_ok: Some(Test1{data: 3})
    };
    println!("{:?}", c1);
    println!("{:?}", c1.trait_ok.unwrap().set(12));
}
```

[playground](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=ef1c26bcc32de6a79962e526da00aff5)

Dalam kasus ini:

```
Test1, Test2 -> Uniform, bernoulli dll
ContainerStruct -> OneDimensionalVectorParams
```

tapi di kodeku kenapa ga bisa ya?
hmmm?

hmmm ternyata penyebabnya ini:

```
pub fn random(&self) -> OneDimensionalVectorParams<T, D>;
```

Kalo fungsinya return `OneDimensionalVectorParams<T, D>`
somehow dia error.

kenapa ya?
kita cari de di google.
ini bro

> The problem arises because you tried to lie to the compiler.
> This code:
>
> ```
> impl<T> Foo<T> {
>    fn new() -> Self {}
> }
> ```
>
> Says "For whatever T the caller chooses, I will create a Foo
> with that type". Then your actual implementation picks a
> concrete type — in the example, a bool. There's no guarantee
> that T is a bool. Note that your new function doesn't even
> accept any parameter of type T, which is highly suspect as
> that's how the caller picks the concrete type 99% of the time.

oke oke.

Show we have the following concept:

```
one_dim() -> vector dimension
with_shape() -> vector shape
zeros(), ones(), full_of(), random() -> vector values

random values have distribution:
1. uniform()
    uniform distribution have range type
    -> closed_irange
    -> half_open_range
2. normal()
    mean
    standard deviation
3. standard normal
2. Cauchy
    median
    scale

Full: https://docs.rs/rand/0.6.5/rand/distributions/index.html
```

Brainstorming API:

Random from uniform distribution
```rust
// Uniform default half-open range [0, 1)
let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .random()
    .uniform()
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .uniform()
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .random(distribution::uniform())
    .generate();

// Uniform with specified half-open range
let a: Vec<f32> = Vec::one_dim()
    .with_shape([10])
    .random()
    .uniform()
    .in_half_open_range(n, m)
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .random(
        distribution::uniform()
        .in_range(n, m)
    )
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([10])
    .uniform()
    .in_half_open_range(n, m)
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([20])
    .random()
    .uniform()
    .in_range(n, m)
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([20])
    .uniform()
    .in_range(n, m)
    .generate();

// Uniform with closed range
let a: Vec<f32> = Vec::one_dim()
    .with_shape([5])
    .random()
    .uniform()
    .in_closed_range(n, m)
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([5])
    .uniform()
    .in_closed_range(n, m)
    .generate();


let a: Vec<f32> = Vec::one_dim()
    .with_shape([5])
    .uniform(0, 1)
    .generate();
```

Random normal

```rust
let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .random()
    .normal()
    .with_mean(x)
    .with_std_dev(x)
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .normal(mean, std_dev)
    .generate();
```

random standard normal

```rust
// Normal default half-open range [0, 1)
let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .standard_normal()
    .generate();
```

random cauchy

```rust
// Cauchy
let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .random()
    .cauchy()
    .with_median(x)
    .with_scale(x)
    .generate();

let a: Vec<f32> = Vec::one_dim()
    .with_shape([0])
    .cauchy(median, scale)
    .generate();
```

anw, we can't do the following:

```rust
let a: Vec<f32> = Vec::cauchy(median, scale)
    .with_shape()
    .generate()

let a: Vec<Vec<f32>> = Vec::cauchy(median, scale)
    .with_shape()
    .generate()
```

It will raise an `conflicting implementation` error for trait cauchy.
let's try.

kenapa ga konsisten sama `zeros()` ya?

```rust
let a: Vec<f32> = Vec::one_dim()
    .with_shape([12])
    .zeros()
    .generate();
```

harusnya kan

```rust
let a: Vec<f32> = Vec::zeros()
    .with_shape([12])
    .generate();
```

hmmmmmm let's just use these consistent and simple API:

```rust
let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .zeros()
    .generate();

let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .ones()
    .generate();

let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .uniform(low, high)
    .generate();

let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .normal(mean, std_dev)
    .generate();

let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .cauchy(median, scale)
    .generate();
// and so on, nice.
```

untuk randomnya, itukan random based on the distribution

```rust
let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .uniform_distribution(low, high)
    .generate();

let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .normal_distribution(mean, std_dev)
    .generate();

let matrix: Vec<Vec<i32>> = Vec::two_dim()
    .with_shape([2, 2])
    .cauchy_distribution(median, scale)
    .generate();
```

How to pronounce?

1. Generate two-dimensional vector with shape [3, 3]
   filled with random samples from a uniform distribution
   over half-open interval [0, 1)
2. Generate two-dimensional vector with shape [3, 3]
   filled with random samples from a normal distribution
   with mean x and standard deviation y

```rust
let matrix: Vec<Vec<f64>> = Vec::two_dim()
    .with_shape([3, 3])
    .random_samples_from_uniform_distribution(low, high)
    .generate();

let matrix: Vec<Vec<f64>> = Vec::two_dim()
    .with_shape([3, 3])
    .uniform_distribution(low, high)
    .generate();
```

I think I'm gonna use

```rust
let matrix: Vec<Vec<f64>> = Vec::two_dim()
    .with_shape([3, 3])
    .uniform_distribution(low, high)
    .generate();
```

keyh

# Multiple Struct Params in One builder
We cannot branch out the builder struct for each method.
It will cause "unconstrained type parameter" error

```
   Compiling crabsformer v2019.3.8 (/Users/pyk/pyk/Crabsformer)
error[E0207]: the type parameter `T` is not constrained by the impl trait, self type, or predicates
  --> src/builders/one_dimensional.rs:85:6
   |
85 | impl<T> OneDimensionalVectorParams
   |      ^ unconstrained type parameter
```

So we can't use proxy like this:

```
                                            /-> zeros() -> OneDimensionalDefaultValueParams<T>{}
one_dim() -> OneDimensionalVectorParams{} -
                                            \-> uniform_distribution() -> OneDimensionalUniformDistributionParams<U>{}
```

First of all, why we use `one_dim()`? We use `one_dim` to specify the dimension
of the vector and we can't use `zeros()` directly for other dimension. We can't do
these:

```rust
let a: Vec<f32> = Vec::zeros();
let a: Vec<Vec<f32>> = Vec::zeros();
```

We can't do overloading: `zeros()` to return `Vec<f32>` and `Vec<Vec<f32>>`.

Btw we can use [Associated types] to this right? I don't know. Let's stry.

[Associated types]: https://doc.rust-lang.org/stable/book/ch19-03-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types

We can use associated types like the following:

```rust
pub trait Zero<T>
where
    T: Num + FromPrimitive + Copy,
{
    type Output;

    fn zeros() -> Self::Output;
}


impl<T> Zero<T> for Vec<T>
where
    T: Num + FromPrimitive + Copy,
{
    type Output = OneDimensionalDefaultValueParams<T>;
    fn zeros() -> Self::Output {
        OneDimensionalDefaultValueParams {
            shape: [1],
            default_value: T::from_f32(0.0).unwrap(),
        }
    }
}

impl<T> Zero<T> for Vec<Vec<T>>
where
    T: Num + FromPrimitive + Copy,
{
    type Output = TwoDimensionalDefaultValueParams<T>;
    fn zeros() -> Self::Output {
        TwoDimensionalDefaultValueParams {
            shape: [1, 2],
            default_value: T::from_f32(0.0).unwrap(),
        }
    }
}
```

But, if we want to call the `zeros()` we need to do
the following:

```rust
let a: Vec<f32> = Vec::<f32>::zeros().generate();
println!("{:?}", a);

let a: Vec<Vec<f32>> = Vec::<Vec<f32>>::zeros().generate();
println!("{:?}", a);
```

otherwise it will raise a compiler error like the following:

```
error[E0282]: type annotations needed
   --> src/main.rs:136:23
    |
136 |     let a: Vec<f32> = Vec::zeros().generate();
    |                       ^^^^^^^^^^^^ cannot infer type
    |
    = note: type must be known at this point
```

hmmm, so associated types doesn't resolve this problem:

```rust
let a: Vec<f32> = Vec::zeros();
let a: Vec<Vec<f32>> = Vec::zeros();
```

why we need `Vec` ? Vector? yup. So we end up use `*_dim` right?

```rust
let a: Vec<f32> = Vec::one_dim().zeros();
let a: Vec<Vec<f32>> = Vec::two_dim().zeros();
```

but we can't do the following:

```rust
let a: Vec<f32> = Vec::one_dim().zeros();
let a: Vec<f32> = Vec::one_dim().uniform_distribution();

let a: Vec<Vec<f32>> = Vec::two_dim().zeros();
let a: Vec<Vec<f32>> = Vec::two_dim().uniform_distribution();
```

because of this

```
                                            /-> zeros() -> OneDimensionalDefaultValueParams<T>{}
one_dim() -> OneDimensionalVectorParams{} -
                                            \-> uniform_distribution() -> OneDimensionalUniformDistributionParams<U>{}
```

`zeros()` and `uniform_distribution()` are requires different type.

hmmmmmmmmmmmmmmmm ...

I think our current approach is wrong...

> A trait tells the Rust compiler about functionality a
> particular type has and can share with other types.
> We can use traits to define shared behavior in an
> abstract way. We can use trait bounds to specify
> that a generic can be any type that has certain behavior.
>
> -- [Rust book](https://doc.rust-lang.org/book/ch10-02-traits.html)

We use trait as a way to build vector, not the other arround.
We need type first.

our type is `Vec<T>`, `Vec<Vec<T>>` and so on.

Fak it. Let's just focus on `Vector<T>` and `Matrix<T>`.


# Vector<T> & Matrix<T>

We define vector and the matrix like the following:

```rust
use num::{FromPrimitive, Num};

#[derive(Debug)]
pub struct Vector<T>
where
    T: FromPrimitive + Num,
{
    size: usize,
    data: Vec<T>,
}

#[derive(Debug)]
pub struct Matrix<T>
where
    T: FromPrimitive + Num,
{
    nrows: usize,
    ncols: usize,
    data: Vec<Vec<T>>,
}
```

Let's prove first how to implement this:

```rust
let a: Vector<f32> = Vector::zeros(size);
let a: Vector<f32> = Vector::ones(size);
let a: Vector<f32> = Vector::full(size, value);
let a: Vector<f32> = Vector::uniform(size, low, high);
let a: Vector<f32> = Vector::normal(size, mean, std_dev);
```

To:

```rust
let a: Vector<f32> = Vector::with_size(size).zeros();
let a: Vector<f32> = Vector::with_size(size).ones();
let a: Vector<f32> = Vector::with_size(size).full(value);
let a: Vector<f32> = Vector::with_size(size).uniform(low, high);
let a: Vector<f32> = Vector::with_size(size).normal(mean, std_dev);
```

Prototype available [here](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=b19da9ed5c99ad0c67f348821259392c)


TODO, we will just storm the the door by following tutorials and implement
it using Crabsformer. Anyway we need to add module datasets. I think it's
very crucial for tutorial?

- We need to find a way to save the dataset to a file
- We need to be able to load the dataset

```rust
use crabsformer::dataset;

dataset::load_mnist("path")
dataset::load_dataset_name("path")
```

# API: Rule of Thumb
We should use constructor only as a static method.
oke.

Rule:

```
static method -> for builder
instance method -> vector ops
```

# Vector operations
https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.htm

Use this guide as an example https://www.pluralsight.com/guides/overview-basic-numpy-operations

# What to do
still follow this https://docs.scipy.org/doc/numpy/user/quickstart.html

lanjut matrix operations, guidenya di numpy yg "Basic Operations"

List of unary operations
https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ndarray.html#numpy.ndarray

# Numeric Vector Slice
I want to support slicing in vector, like the following:

```rust
let x = vector![3, 1, 4, 1];

// Indexing
assert_eq!(x[0], 3);
assert_eq!(x[2], 4);

// Slicing
assert_eq!(x[0..2], vector![3, 1, 4]);
assert_eq!(x[2..], vector![4, 1]);
assert_eq!(x[..1], vector![3, 1]);
```

In order to support that, I need to implement `ops::Index<ops::Range<usize>>`
for `Vector<T>`.

# Slice
So, any other sequence-like data type have a data type conterpart
that represents a borrowed value. For example:

```
owned: String -> borrowed: str
owned: CString -> borrowed: CStr
owned: Vec<T> -> borrowed: &[T]
owned: PathBuf -> borrowed: path
```

nah bagaimana cara buat borrowed version dari `Vector<T>` ?

Wait we should look at how `Vec<T>` implement the Index trait
first.

Ok, the borrowed counterpart is only a struct.

But how? We can't return created struct as reference right?

Ok, this is work:

```rust
use std::ops;

// Owned & Borrowed value
#[derive(Debug)]
struct Vector<T> {
    elements: Vec<T>
}

impl<T> ops::Index<ops::Range<usize>> for Vector<T>
where
    T: Copy,
{
    type Output = [T];

    fn index(&self, index: ops::Range<usize>) -> &[T] {
        &self.elements[index]
    }
}

fn main() {
    let a = Vector{elements: vec![1, 2, 3, 4] };
    println!("a = {:?}", &a[1..4])
}
```

but, we can't perform operation on them?
should we create a `SubVector` ?
yes.

Anw, let's just use `&[T]` numeric slice.
Oke.

No no, we can't use `&[T]` as numeric slice.
we can't do like the following:

```rust
impl<T> ops::Add<&[T]> for &[T] {
    type Output = [T];
}
```

it will raise an error like the following:

```
type parameter `T` must be used as the type parameter
for some local type (e.g. `MyStruct<T>`)

type parameter `T` must be used as the type
parameter for some local type

note: only traits defined in the current crate can be
 implemented for a type parameter rustc(E0210)
vector.rs(231, 1): type parameter `T` must be used as the
type parameter for some local type
```

We need to create some kind of `VectorSlice` or
`SubVector`.

Do we need to do arithmetic operation on slice?

We can create trait called slice and implement it.
Fak, this is easy. Keyh.

So we can do like the following:

```rust
let x = vector![3, 1, 2, 3];

// Range
assert_eq!(x.slice(0..1), vector![3]);

// RangeTo
assert_eq!(x.slice(..2), vector![3, 1]);

// RangeFrom
assert_eq!(x.slice(2..), vector![2, 3]);

// RangeFull
assert_eq!(x.slice(..), vector![3, 1, 2, 3]);

// RangeInclusive
assert_eq!(x.slice(0..=1), vector![3, 1]);

// RangeToInclusive
assert_eq!(x.slice(..=2), vector![3, 1, 2]);
```

keyh.

# Guides
1. Numerical calculations with NumPy http://kestrel.nmt.edu/~raymond/software/python_notes/paper003.html

# Why We Don't Need to Implement Index trait in Matrix?

> In my eyes, you just gave a good reason to disallow using Index for a Matrix.
> It's very, very unclear as to whether Index would return a row or a column.
> Why not have row and column functions?

Reddit discussion: [here](https://www.reddit.com/r/rust/comments/4as7gx/why_make_the_index_trait_so_useless/)

# Get Column from Row Major Matrix
Here link to the [playground](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=ff1aa666b75e5774b3e8c1f989e4f078

# SubMatrix
This is a Vector and SubVector:

```rust
pub struct Vector<T>
where
    T: Num + Copy,
{
    elements: Vec<T>,
}

pub struct SubVector<'a, T>
where
    T: Num + Copy,
{
    inner: &'a [T],
}
```

How Matrix and SubMatrix ? like this?

```rust
#[derive(Debug)]
pub struct Matrix<T>
where
    T: Num + Copy,
{
    /// Matrix size
    nrows: usize,
    ncols: usize,
    vec: Vector<T>,
}

#[derive(Debug)]
pub struct SubMatrix<'a, T>
where
    T: Num + Copy,
{
    nrows: usize,
    ncols: usize,
    vec: SubVector<'a, T>,
}
```

Masalahnya adalah kalau row matrix atau matrix 1xm itu bisa.
masalahnya ketika matrix mx1 atau mxn. SubMatrix berarti tidak contiguous
lagi ya.

```rust
#[derive(Debug)]
pub struct SubMatrix<'a, T>
where
    T: Num + Copy,
{
    nrows: usize,
    ncols: usize,
    vec: Vec<SubVector<'a, T>>,
}
```

Jadi kek di store per rows gitu, soalnya satu rows di vecnya adalah SubVector<'a, T>
keyh.

Masalah 1: Cara compare SubMatrix dengan Matrix.

SubMatrix adalah Vector of SubVector.
kenapa? karena waktu slice dapetnya cuman itu. kalo di flatten gmn?
Jadi SubVector<'a, T> ?
apakah bisa?


# Sampai mana?
Indexing Submatrix.

# Submatrix, Row matrix, Column matrix
Anw, we see submatrix, row matrix, column matrix as a way to see the data. The only contains
index information such as position, offset, size and the reference to the original matrix.

We need to treat Subvector like this as well.
