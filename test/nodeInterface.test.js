"use strict";

var base = require('./_base'),
  sinon = base.sinon,
  assert = base.assert,
  expect = base.expect,
  should = base.should;


var ml = require('../index');

var test = module.exports = {};

test['node interface'] = function() {
  ml.should.eql(require('../dist/machine-learning'));
};
