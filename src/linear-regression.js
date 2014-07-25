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




