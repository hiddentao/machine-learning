"use strict";

var _ = require('lodash'),
  base = require('./_base'),
  sinon = base.sinon,
  assert = base.assert,
  expect = base.expect,
  should = base.should;


var mocker, ml, X, y;


var test = module.exports = {
  beforeEach: function() {
    mocker = sinon.sandbox.create();
    ml = require('../index')();
  },
  afterEach: function() {
    mocker.restore();
  },
  'gradient descent ': {
    'basic': {
      beforeEach: function() {
        X = new ml.Matrix([ [34, 23], [20, 11], [41, 10], [54, 12] ]);
        y = new ml.Matrix([ [0.9], [1.1], [2.2], [0.8] ]);
      },

      'returns mean and std': function() {
        var costFn = mocker.spy(function(X, theta, y) {
          return 234.2344;
        });

        var ret = ml.gradientDescent(X, y, costFn, 0.1, 0);

        ret.mean.data.should.eql([ [ 37.25, 14 ] ]);
        base.round(ret.std.data[0]).should.eql([ 14.1745, 6.0553 ]);
      },

      '0 iterations - initial cost': function() {
        var costFn = mocker.spy(function(X, theta, y) {
          return 234.2344;
        });

        var ret = ml.gradientDescent(X, y, costFn, 0.1, 0);

        ret.cost.should.eql( 234.2344 );
        costFn.should.have.been.calledOnce;
        var costArgs = costFn.getCall(0).args;
        costArgs[0].data.should.eql( ml.normalizeFeatures(X).X.data );
        costArgs[1].should.eql(ret.theta);
        costArgs[2].should.eql(y);

        ret.theta.toArray().should.eql([ [0], [0], [0] ]);
        ret.alpha.should.eql(0.1);
        ret.iters.should.eq(0);
      },

      'after 1 iteration': function() {
        var costFn = mocker.spy(function(X, theta, y) {
          return 234.2344;
        });

        var ret = ml.gradientDescent(X, y, costFn, 0.1, 1);

        costFn.should.have.been.calledTwice;
        ret.theta.toArray().should.eql([ [0.125], [-0.0004409324232894696], [-0.023120239067653578] ]);
        ret.alpha.should.eql(0.1);
        ret.iters.should.eq(1);
      },

      'exhausts iterations': function() {
        var costFn = mocker.spy(function(X, theta, y) {
          return 234.2344;
        });

        var ret = ml.gradientDescent(X, y, costFn, 0.1, 10);

        costFn.callCount.should.eql(11);
        ret.alpha.should.eql(0.1);
        ret.iters.should.eq(10);
      },

      'halves alpha if cost goes up': function() {
        var called = 0;
        var costFn = mocker.spy(function(X, theta, y) {
          called++;
          if (1 < called) {
            return 50;
          } else {
            return 20;
          }
        });

        var ret = ml.gradientDescent(X, y, costFn, 0.1, 5);

        costFn.callCount.should.eql(6);
        ret.alpha.should.eql(0.003125);
        ret.iters.should.eq(5);
      },
    },

    'real data': {
      'linear regression': {
        'm=97, n=1': function() {
          var data = base.loadLinRegData('linreg-1f.txt'),
            X = new ml.Matrix(data.X),
            y = new ml.Matrix(data.y);

          var ret = ml.gradientDescent(X, y, ml.LinReg.costFunction, 0.01, 1500);

          var theta = base.round(_.flatten(ret.theta.toArray()));

          theta.should.eql([ 5.8391, 4.6169 ]);
        },
        'm=97, n=2': function() {
          var data = base.loadLinRegData('linreg-2f.txt'),
            X = new ml.Matrix(data.X),
            y = new ml.Matrix(data.y);

          var ret = ml.gradientDescent(X, y, ml.LinReg.costFunction, 0.1, 400);

          var theta = base.round(_.flatten(ret.theta.toArray()));

          theta.should.eql([ 340412.6596, 110631.0467, -6649.4707 ]);
        }
      }
    }
  }
};



