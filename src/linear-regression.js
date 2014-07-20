/**
 * Construct a new linear regression solver.
 *
 * @param {Integer} numFeatures The number of features in the training set.
 * 
 * @constructor
 */
ML.LinearReg = function(numFeatures) {
  this._dataX = [];
  this._dataY = [];
  this._nF = numFeatures;   // need extra space to hold x0
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
ML.LinearReg.prototype.addData = function(data) {
  var rows = data.length;

  for (var i=0; i<data.length; ++i) {
    this._dataX.push(data[i].slice(0, rows-1));
    this._dataY.push(data[i][rows-1]);
  }
};




/**
 * Solve this regression using gradient descent.
 *
 * @param {Number} alpha Initial learning rate.
 * @param {Integer} maxIters Max. no. of iterations to perform.
 * 
 * @return {Object} {theta: Vector, cost: float, alpha: float, iters: int}
 */
ML.LinearReg.prototype.solve = function(alpha, maxIters) {
  var X = new ML.Matrix(this._dataX),
    y = new ML.Vector(this._dataY);

  return ML.gradientDescent(X, y, ML.LinearReg.costFunction);
};






/**
 * The cost function.
 * @return {Number}
 */
ML.LinearReg.costFunction = function(X, theta, y) {
  /*
    m = y.size
   
    Cost = 1/2m * trans(XT*theta - y) * (XT*theta - y), where XT=trans(X)
  */
  var X_mul_theta_minus_y = X.mul(theta).minusP(y);

  return (X_mul_theta_minus_y.dot(X_mul_theta_minus_y)) / (2 * y.size);
};




