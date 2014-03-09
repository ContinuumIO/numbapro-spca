%-------------------------------------------------------------------------------
%% Generate an example input for sparse PCA
%% Assume that A is given to you; This is only for my demo
p = 1000; 		% number of features per sample
n = 100;			% number of samples

X = randn(n, p);	% Create random data

A = (1/n)*X'*X;	
[U, S, ~] = svd(A); 
A = U * ( S*diag(1./[1:p]) ) * U'; % this is because I want A to behave like
  								  % real input data do. Don't worry about this part
clear p n X U S
%-------------------------------------------------------------------------------

%% Set up
% Input A, epsilon, d, k

epsilon = 0.1;
d = 3;
k = 10;		% must be between 1 and p, (p is the number of features in the data)
[p, ~] = size(A);

%% Prepare input
[U, S, ~] = svd(A); 
					% This is like [U, S] = eig(A), (A is a PSD matrix).
					% Here I use SVD because this is way more stable 
					% numerically.
					% It produces all n eigenvectors of A.
					% But we do not need them all. We only need the top d.
					% By the way, the eig function is the part that Alex was 
					% saying can be done with the so called power-iteration 
					% method, which involves multiplying a matrix with a vector
					% several times. We can talk about it...

% Sort the eigenvalues to determine which are the d largest ones

% Keep the d columns of U that correspond to the d leading eigenvectors,
% scaled with the squared root of the corresponding eigenvalues.

Vd = U(:, 1:d) * S(1:d, 1:d).^0.5; % p x d matrix

numSamples = (4/epsilon)^d; % Determine number of samples

%-------------------------------------------------------------------------------
%% The actual alogorithm

opt_x = zeros(p, 1);
opt_v = -Inf;

for i = 1:numSamples  % Each iteration is independent from the previous

	c = randn(d, 1); % Generate a random d-dimensional vector
	c = c / norm(c); % normalize its length: divide each entry by Sum(c_i^2);	
	a = Vd*c; 		 % This is a long vector with dimension p
					 % "a" is essentically a random vector in the range of Vd.
				
	[~, I] = sort(abs(a), 'descend');	% I(1:k) contains the indices of the
										% k largest entries of a (by abs value)
	
	val = norm( a(I(1:k)) );

	if val > opt_v

		opt_v = val;

		% create opt_x using the location and values of the k largest (by
		% absolute value) entries of a.
		opt_x(:)= 0;
		opt_x(I(1:k)) = a(I(1:k)) / val;
	end

end

% Although in a serial implementation this would pointless, I could essentially
% create a vector x in each iteration, store all x's, then compute 
% norm(Vd'*x) for all of them, and keep the best x (the one achieving
% the highest norm);

% opt_x is what we are interested in
% We are interested in maximizing x'*A*x. 
% opt_x should be approximately the best x
% Also, norm(Vd'*opt_x) should be less than and hopefully close to opt_x'*A*opt_x
norm(Vd'*opt_x)
opt_x'*A*opt_x
