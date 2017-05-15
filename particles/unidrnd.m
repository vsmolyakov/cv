function r = unidrnd(n,mm,nn)

rows = mm;
columns = nn;

r = ceil(n .* rand(rows,columns));

% Fill in elements corresponding to illegal parameter values
if prod(size(n)) > 1
    r(n < 0 | round(n) ~= n) = NaN;
elseif n < 0 | round(n) ~= n
    r(:) = NaN;
end
