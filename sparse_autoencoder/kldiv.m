function kl = kldiv(a, b)
kl = a .* log(a./b) + (1-a) .* log((1-a) ./ (1-b));

