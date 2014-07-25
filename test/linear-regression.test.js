"use strict";

var base = require('./_base'),
  sinon = base.sinon,
  assert = base.assert,
  expect = base.expect,
  should = base.should;


var mocker, ml;


var test = module.exports = {
  beforeEach: function() {
    mocker = sinon.sandbox.create();
    ml = require('../index')();
  },
  afterEach: function() {
    mocker.restore();
  },
  'linear regression': {
    'constructor': function() {
      var l = new ml.LinReg(5);

      l._dataX.should.eql([]);
      l._dataY.should.eql([]);
      l._nF.should.eql(5);
    },
    'add data': {
      'checks for enough data': function() {
        var l = new ml.LinReg(5);

        expect(function() {
          l.addData([
            [23, 45, 12, 98, 19],
          ]);          
        }).to.throw('machine-learning: [LinReg_addData] Not enough data');

      },
      'adds to internal arrays': function() {
        var l = new ml.LinReg(5);

        // the extra values in this array should get ignored
        l.addData([
          [23, 45, 12, 98, 34, 18, 81],
          [87, 48, 9, 1, 45, 23, 72]
        ]);

        l.addData([
          [8, 7, 6, 5, 3, 9],
        ]);

        l._dataX.should.eql([
          [23, 45, 12, 98, 34],
          [87, 48, 9, 1, 45],
          [8, 7, 6, 5, 3]
        ]);

        l._dataY.should.eql([ 18, 23, 9 ]);
      }
    },
    'solve': function() {
      var spy = mocker.stub(ml, 'gradientDescent').returns(234234);

      var l = new ml.LinReg(5);

      l.addData([
        [23, 45, 12, 98, 34, 12],
        [87, 48, 9, 1, 45, 9]
      ]);

      l.solve().should.eql(234234);

      l._results.should.eql(234234);

      spy.should.have.been.calledOnce;
      var args = spy.getCall(0).args;

      args[0].should.be.instanceOf(ml.Matrix);
      args[0].data.should.eql([
        [23, 45, 12, 98, 34],
        [87, 48, 9, 1, 45]
      ]);

      args[1].should.be.instanceOf(ml.Matrix);
      args[1].data[0][0].should.eql(12);
      args[1].data[0][1].should.eql(9);

      args[2].should.eql(ml.LinReg.costFunction);
    },
    'calculate': {
      'need results first': function() {
        var l = new ml.LinReg(5);

        expect(function() {
          l.calculate();
        }).to.throw('machine-learning: [LinReg_calculate] Need to solve first');
      },
      'calculates result': function() {
        var l = new ml.LinReg(3);

        l._results = {
          theta: new ml.Matrix([ [0.5], [1.5], [2.5] ]),
          mean: new ml.Matrix([ [5, 10, 15] ]),
          std: new ml.Matrix([ [2, 1, 3] ]),
        };

        var v = l.calculate([8, 7, 24]);

        var exp = new ml.Matrix([1.5, -3, 3]).dot(l._results.theta).data[0][0];

        v.should.eql(exp);
      }
    },
    'cost function': function() {
      var X = new ml.Matrix([
        [1, 2, 3],
        [1, 5, 6]
      ]);
      var theta = new ml.Matrix([0.5, 0.3, 0.5]).trans_();
      var y = new ml.Matrix([1, 2]).trans_();

      var cost = ml.LinReg.costFunction(X, theta, y);

      cost.should.eql(2.89);
    }
  }
};



