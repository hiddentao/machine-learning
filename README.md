# machine-learning

**THIS LIBRARY IS NOT YET RELEASED. STILL IN DEVELOPMENT...**

[![Build Status](https://secure.travis-ci.org/hiddentao/machine-learning.png?branch=master)](http://travis-ci.org/hiddentao/machine-learning)

Efficient, high-performance machine-learning algorithms.

This implements the various machine learning algorithms as covered by the Coursera ML course, and is intended for use in node.js and the browser.

I recommend against using this library in your main execution thread as some of these algorithms take time to run, even with small datasets. In the browser you ought to run these algorithms within a [WebWorker](http://www.w3.org/TR/workers/). In node.js there are many ["worker" mechanisms](https://www.npmjs.org/search?q=webworker) you can use.

Features:

* Uses [linear-algebra](https://github.com/hiddentao/linear-algebra) for high-performance calculations.
* Comprehensive unit tests.
* Works in node.js and in browsers.
* TODO...

## Installation

### node.js

Install using [npm](http://npmjs.org/):

    $ npm install machine-learning

To use it:

```js
var ml = require('machine-learning')();
// Ready!
```

### Browser

Use [bower](https://github.com/bower/bower):

    $ bower install machine-learning  # this will also install the linear-algebra package

To use it:

```html
<script type="text/javascript" src="linear-algebra.min.js" />
<script type="text/javascript" src="machine-learning.min.js" />
<script type="text/javascript">
var ml = machineLearning();
// Ready!
</script>
```

## Linear algebra library

This uses the [linear-algebra](https://github.com/hiddentao/linear-algebra) library to perform all the big calculations. However you can specify a different linear-algebra library (which conform to the same API) during initialization:

```js
// Node.js
var MyLinearAlgebraLib = ...
var ml = require('machine-learning')(MyLinearAlgebraLib);

// Browser
var MyLinearAlgebraLib = ...
var ml = machineLearning(MyLinearAlgebraLib);
```

## Examples

## Building

To build the code and run the tests:

    $ npm install -g gulp
    $ npm install
    $ npm test

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/hiddentao/machine-learning/blob/master/CONTRIBUTING.md).

## License

MIT - see [LICENSE.md](https://github.com/hiddentao/machine-learning/blob/master/LICENSE.md)