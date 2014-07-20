var gulp = require('gulp'),
  path = require('path');

var concat = require('gulp-concat');
var jshint = require('gulp-jshint');
var uglify = require('gulp-uglify');
var expect = require('gulp-expect-file');
var mocha = require('gulp-mocha');
var runSequence = require('run-sequence');


gulp.task('build-lib', function() {
  return gulp.src( [
      './src/_header.js',
      './src/normalization.js',
      './src/gradient-descent.js',
      './src/linear-regression.js',
      './src/_footer.js'
    ] )
    .pipe( concat('machine-learning.js') )
    .pipe( gulp.dest('./dist') )
    ;
});


gulp.task('jshint', ['build-lib'], function() {
  return gulp.src([
        './dist/machine-learning.js', 
    ])
    .pipe(jshint())
    .pipe(jshint.reporter('default'))
    .pipe(jshint.reporter('fail'))
  ;
});


gulp.task('minify-lib', ['jshint'], function() {
  return gulp.src('./dist/machine-learning.js')
    .pipe(concat('machine-learning.min.js'))
    .pipe(uglify())
    .pipe( gulp.dest('./dist') )
  ;
});


gulp.task('js', ['minify-lib']);


gulp.task('verify-js', function() {
  return gulp.src( path.join('./dist/*.js') )
    .pipe( expect([
      'dist/machine-learning.js',
      'dist/machine-learning.min.js',
    ]) )
  ;
})



gulp.task('test', function () {
  return gulp.src('./test/*.test.js', { read: false })
      .pipe(mocha({
        ui: 'exports',
        reporter: 'spec'
      }))
    ;
});




gulp.task('default', function(cb) {
  runSequence('js', 'verify-js', 'test', cb);
});



