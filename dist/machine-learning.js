(function (root, factory) {
  "use strict";

  // AMD
  if (typeof define === 'function' && define.amd) {
    define(['linear-algebra'], factory);
  }
  // CommonJS
  else if (typeof exports === 'object') {
    module.exports = factory(require('linear-algebra'));
  }
  // Browser
  else {
    root.machineLearning = factory(root.linearAlgebra);
  }
})(this, function (defaultLinearAlgebra) {
  "use strict";


  var _SMALLEST_DELTA = Number.EPSILON || 0.000000000001;


  var _throwError = function(cat, op, msg) {
    throw new Error('machine-learning: [' + cat + '_' + op + '] ' + msg);
  };


  /**
   * Initialise the machine learning library.
   *
   * @param {Object} linearAlgebra Linear algebra library.
   * 
   * @return {Object} Initialized machine learning algorithms.
   */
  return function(linearAlgebra) {
    if (!linearAlgebra) {
      linearAlgebra = defaultLinearAlgebra();
    }

    // namespace
    var ML = {
      Matrix: linearAlgebra.Matrix,
      Vector: linearAlgebra.Vector
    };


    
/**
 * Get normalized features array.
 *
 * @param {Matrix} X training features dataset.
 * 
 * @return {Object} { X: Normalized version of training set with x0 column added., mean: Matrix of column means, std: Matrix of column std }
 */
ML.normalizeFeatures = function(X) {
  var numFeatures = X.cols,
    m = X.rows;

  // for each value in each column, subtract column mean from it 
  // and divide by result by column std (standard deviation)

  var newArray = new Array(m);
  
  var mean = new Array(numFeatures), std = new Array(numFeatures);

  var i, j, jNormalized, tmpSum = new ML.Vector.zero(m);

  for (j=0; j<numFeatures; ++j) {
    jNormalized = j + 1;  // because we'll be prepending an x0 column later on

    // calculate mean
    for (i=0; i<m; ++i) {
      tmpSum.data[0][i] = X.data[i][j];

      // pre-allocate final array with space for x0 column
      if (!Array.isArray(newArray[i])) {
        newArray[i] = new Array(numFeatures + 1); // with space for x0 column
        newArray[i][0] = 1; // x0 column
      }
    }
    mean[j] = tmpSum.getSum() / m;

    // calculate std
    for (i=0; i<m; ++i) {
      // subtract each column value from the mean
      newArray[i][jNormalized] = X.data[i][j] - mean[j];
      // and calculate std at the same time
      tmpSum.data[0][i] = newArray[i][jNormalized] * newArray[i][jNormalized];
    }
    std[j] = Math.sqrt(tmpSum.getSum() / (m-1));   // m-1 for unbiased estimation (Bessel's correction)
    if (0 === std[j]) { 
      std[j] = _SMALLEST_DELTA; // to prevent divide by zero below
    }

    // divide column by std
    for (i=0; i<m; ++i) {
      // subtract each column value from the mean
      newArray[i][jNormalized] /= std[j];
    }
  }

  return {
    X: new ML.Matrix(newArray),
    mean: new ML.Matrix(mean),
    std: new ML.Matrix(std),
  };
};



/**
 * Perform gradient descent to compute the value of theta.
 *
 * This first calls `normalizeFeatures()` to normalize the dataset, so that 
 * gradient descent can converge quicker.
 * 
 * Once complete this returns the final `theta`, `cost`, the final learning 
 * rate (`alpha`) as well as the number of iterations elpased (`iter`).
 *
 * @param {Matrix} X Training feature data size m x n (without x0 column)
 * @param {Vector} y Training result data, size m x 1
 * @param {Function} costFn Function to compute cost.
 * @param {Number} alpha Initial learning rate.
 * @param {Integer} maxIters Max. no. of iterations to perform.
 * 
 * @return {Object} {theta: Matrix(n+1 x 1), cost: float, alpha: float, iters: int, mean: Matrix(1xn), std: Matrix(1xn) }
 */
ML.gradientDescent = function(X, y, costFn, alpha, maxIters) {
  /*
    1. Feature-normalization
    2. Initialise theta to be a zero-vector
    3. For each iteration:    
      3.0 If cost is 0 or maxIters reached then goto 4
      3.1 Calculate new theta, where:
        theta(j) = theta(j) - (alpha / m) * {i=1..m: (X(i)*theta - y(i))*(X(i)(j)) }
      3.2 Update theta
      3.3 Calculate new cost
      3.4 If cost increased then alpha = alpha / 2
    4. Exit and return theta, cost, alpha, iters
  */
  
  // the features array 
  var _norm = ML.normalizeFeatures(X);
  X = _norm.X;

  // measurements
  // (remember, at this point X has the x0 column prepended)
  var m = X.rows,
    n = X.cols;

  var i, j, row, h_x, tmpSum = new ML.Vector.zero(m).trans_();

  // initial theta and theta delta as column vectors
  var theta = ML.Vector.zero(n).trans_(),
      delta = ML.Vector.zero(n).trans_();

  // initial cost
  var cost = costFn(X, theta, y),
    oldCost = cost;

  // loop
  var iters = 0;
  while (0 < cost && maxIters > iters) {
    iters++;

    // calculate h(x) - y
    h_x = X.dot(theta).minus_(y);

    // for each theta value
    for (j=0; j<n; ++j) {
      // calculate sum = (h(x(i)) − y(i))*x(i,j)
      for (i=0; i<m; ++i) {
        tmpSum.data[i][0] = h_x.data[i][0] * X.data[i][j];
      }

      // delta = alpha * (sum / m)
      delta.data[j][0] = alpha * tmpSum.getSum() / m;
    }

    // update theta
    theta.minus_(delta);

    // calculate new cost
    cost = costFn(X, theta, y);

    // if cost increased
    if (cost > oldCost) {
      // restore theta
      theta.plus_(delta);
      // reduce alpha
      alpha /= 2;
    } else {
      // save current cost
      oldCost = cost;
    }
  }

  return {
    theta: theta,
    cost: cost,
    alpha: alpha,
    iters: iters,
    mean: _norm.mean,
    std: _norm.std,
  };
};



/**
 * Construct a new linear regression solver.
 *
 * @param {Integer} numFeatures The number of features in the training set.
 * 
 * @constructor
 */
ML.LinReg = function(numFeatures) {
  this._dataX = [];
  this._dataY = [];
  this._nF = numFeatures;
};




/**
 * Add data to the training set.
 *
 * Each row in the training set is an Array consisting of the `x` values 
 * (the features) followed by the `y` value (the output) at the end.
 *
 * The no. of features is expected to match the value passed in during 
 * construction.
 * 
 * @param  {Array} newTrainingData One or more rows (Array) of training data.
 * @return this
 */
ML.LinReg.prototype.addData = function(data) {
  var rows = data.length;

  for (var i=0; i<rows; ++i) {
    if (data[i].length <= this._nF) {
      _throwError('LinReg', 'addData', 'Not enough data');
    }

    this._dataX.push(data[i].slice(0, this._nF));
    this._dataY.push(data[i][this._nF]);
  }
};




/**
 * Solve this regression using gradient descent.
 *
 * @param {Number} alpha Initial learning rate.
 * @param {Integer} maxIters Max. no. of iterations to perform.
 *
 * @return See gradientDescent()
 */
ML.LinReg.prototype.solve = function(alpha, maxIters) {
  var X = new ML.Matrix(this._dataX),
    y = new ML.Matrix(this._dataY).trans_();

  return (this._results = ML.gradientDescent(X, y, ML.LinReg.costFunction));
};




/**
 * Calculate output for given input.
 *
 * The regression needs to have been solved prior to calling this.
 * 
 * @param {Array} input Containing n items, n = no. of of features
 *
 * @return Number The calculated `y` value
 */
ML.LinReg.prototype.calculate = function(input) {
  if (!this._results) {
    _throwError('LinReg', 'calculate', 'Need to solve first');
  }

  var normalized = 
    new ML.Matrix(input)
      .minus_(this._results.mean)
      .div_(this._results.std);

  return normalized.dot_(this._results.theta).data[0][0];
};






/**
 * The cost function.
 *
 * @param {Matrix} X Size m x n
 * @param {Matrix} theta Size n x 1
 * @param {Matrix} y Size m x 1
 * 
 * @return {Number} Cost using theta
 */
ML.LinReg.costFunction = function(X, theta, y) {
  /*
    Cost = (X*theta - y)^2 * (m / 2)
  */
  var m = y.rows;

  var tmp = X.dot(theta).minus_(y);

  return (tmp.mul(tmp)).getSum() / (2 * m);
};







    return ML;
  };
});

