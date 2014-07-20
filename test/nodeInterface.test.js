"use strict";

var sinon = require('sinon'),
  chai = require('chai'),
  expect = chai.expect,
  should = chai.should();
chai.use(require('sinon-chai'));


var ml = require('../index');

var test = module.exports = {};

test['node interface'] = function() {
  ml.should.eql(require('../dist/machine-learning'));
};
