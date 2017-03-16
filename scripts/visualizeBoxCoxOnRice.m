s = [0, 2, 4, 8, 16];
sigma = 1;
lambda = 2/3;

x = linspace(0, 20, 500);
y = (x.^lambda - 1)/lambda;
yinv = (lambda*y + 1).^(1/lambda);
jac = (lambda*y + 1).^(1/lambda - 1);

xpdf = zeros(length(s), length(x));
ypdf = zeros(length(s), length(y));

for i = 1:length(s)
    pd = makedist('Rician', 's', s(i), 'sigma', sigma);
    xpdf(i, :) = pd.pdf(x);
    ypdf(i, :) = pd.pdf(yinv).*jac;
end

figure(1)
plot(x, xpdf,'-', 'LineWidth', 2)
title('Original')

figure(2)
clf
plot(y, ypdf, '--', 'LineWidth', 2)
title('Box-Cox transformed, \lambda = 2/3')
