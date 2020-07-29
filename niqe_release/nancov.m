## Copyright (C) 1995, 1996, 1997, 1998, 1999, 2000, 2002, 2004, 2005,
##               2006, 2007 Kurt Hornik
## Copyright (C) 2009 Soren Hauberg, Jaroslav Hajek
##
## This file is part of Octave.
##
## Octave is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## Octave is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} nancov (@var{x}, @var{y})
## Compute covariance.
##
## If each row of @var{x} and @var{y} is an observation and each column is
## a variable, the (@var{i}, @var{j})-th entry of
## @code{nancov (@var{x}, @var{y})} is the covariance between the @var{i}-th
## variable in @var{x} and the @var{j}-th variable in @var{y}.
## @iftex
## @tex
## $$
## \sigma_{ij} = {1 \over N-1} \sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})
## $$
## where $\bar{x}$ and $\bar{y}$ are the mean values of $x$ and $y$.
## @end tex
## @end iftex
## If called with one argument, compute @code{cov (@var{x}, @var{x})}.
## @end deftypefn

function c = nancov (x, y, method = "all")

  if (nargin < 1 || nargin > 3)
    print_usage ();
  endif
  
  if (nargin == 1)
    two_inputs = false;
  elseif (nargin == 2 && ischar (y))
    method = y;
    two_inputs = false;
  else
    two_inputs = true;
  endif
  
  if (! ischar (method))
    error ("nancov: method must be a string");
  endif
  
  if (rows (x) == 1)
    x = x.';
  endif
  n = rows (x);
  if (two_inputs)
    if (rows (y) == 1)
      y = y.';
    endif
    if (rows (y) != n)
      error ("nancov: x and y must have the same number of observations");
    endif
  endif

  if (n == 0)
    if (two_inputs)
      c = NA (columns (x), columns (y));
    else
      c = NA (columns (x), columns (x));
    endif
  endif

  switch (lower (method))
    case "all"
      if (two_inputs)
        x = x - ones (n, 1) * sum (x) / n;
        y = y - ones (n, 1) * sum (y) / n;
        c = conj (x' * y) / max (1, n - 1);
      else
        x = x - ones (n, 1) * sum (x) / n;
        c = conj (x' * x) / max (1, n - 1);
      endif
    case "complete"
      ## we simply remove all incomplete rows.
      if (two_inputs)
	r = any (isna (x), 2) | any (isna (y), 2);
        x (r, :) = [];
        y (r, :) = [];
        c = cov (x, y);
      else
        r = any (isna (x), 2);
        x (r, :) = [];
        c = cov (x);
      endif
    case "pairs"
      ## this is the most complicated case.
      if (two_inputs)
        ## save NA masks.
        xnamsk = ! isna (x);
        ynamsk = ! isna (y);
	## set everything non-finite to zero, to avoid Inf*0 and NaN*0
	## products getting in our way.
	xmsk = isfinite (x);
	ymsk = isfinite (y);
	x(! xmsk) = 0;
	y(! ymsk) = 0;
	## means
	mx = sum (x) ./ sum (xmsk);
	my = sum (y) ./ sum (ymsk);
	## subtract them
	x -= ones (n, 1) * mx;
	y -= ones (n, 1) * my;
	## calculate products
	c = conj (x' * y);
	## calc symbolic products
	c1 = xmsk.' * ymsk;
	## scale to get covariances
	c = c ./ max (c1 - 1, 1);
	## calc updated symbolic products
	c2 = xnamsk.' * ynamsk;
	## set the violated elements to NaN
	c(c2 > c1) = NaN; 
	## set the zero-length covs to NA
	c(c2 == 0) = NA;
      else
	## do the same for a single input.
        ## save NA masks.
        xnamsk = ! isna (x);
	## set everything non-finite to zero, to avoid Inf*0 and NaN*0
	## products getting in our way.
	xmsk = isfinite (x);
	x(! xmsk) = 0;
	## means
	mx = sum (x) ./ sum (xmsk);
	## subtract them
	x -= ones (n, 1) * mx;
	## calculate products
	c = conj (x' * x);
	## calc symbolic products
	c1 = xmsk.' * xmsk;
	## scale to get covariances
	c = c ./ max (c1 - 1, 1);
	## calc updated symbolic products
	c2 = xnamsk.' * xnamsk;
	## set the violated elements to NaN
	c(c2 > c1) = NaN; 
	## set the zero-length covs to NA
	c(c2 == 0) = NA;
      endif
  endswitch

endfunction

%!test
%! x = rand (10);
%! cx1 = nancov (x);
%! cx2 = nancov (x, x);
%! assert(size (cx1) == [10, 10] && size (cx2) == [10, 10] && norm(cx1-cx2) < 1e1*eps);

%!error nancov ();

%!error nancov (1, 2, 3);
