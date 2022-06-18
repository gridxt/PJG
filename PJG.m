clc;
clear;


m = 1000; 
A = gallery('poisson',m); % returns (mxm)*(mxm) matrix
A = A + diag(diag(A).*1.0); % make the matrix A diagonally dominant

% RHS Vector
b = (1:size(A,1))';

% Settings
maxNumCompThreads(4); % number of CPU cores to utilize
pjg_block_size = round(size(A,1)/10); % PJG block size: between 1 to n (here n is mxm). 1: Gauss-Seidel, n: Jacobi
max_iter = 1500; % max number of iteration
tol = 10^-6; % tolerance
x0 = zeros(size(A,1),1); % initial guess 


% **** Solve using CPU ****
[x,iter,residual,elapsed_time] = PJG_CPU(A,b,max_iter,tol,x0,pjg_block_size);
if iter>max_iter
    fprintf("PJG method did not converge in %i iterations! Residual = %f \n", iter, residual);
else
    fprintf("PJG method converged in %i iterations! Residual = %e and elapsed time = %f Seconds\n", iter, residual, elapsed_time);
end


% **** Solve using GPU ***
% Uncomment this section if you want to solve the linear system on a GPU device
% gpuDevice(1);
% [x,iter,residual,elapsed_time] = PJG_GPU(transpose(A),b,max_iter,tol,x0,pjg_block_size);
% if iter>max_iter
%     fprintf("PJG method did not converge in %i iterations! Residual = %f \n", iter, residual);
% else
%     fprintf("PJG method converged in %i iterations! Residual = %e and elapsed time = %f Seconds\n", iter, residual, elapsed_time);
% end




% *********** PJG Implementation for Multicore CPUs **********

function [x, iter, res, elapsed]= PJG_CPU(A,b,maxiter,tol,x0,psize)

n = size(A,1);
if psize>n 
    psize = n; % Jacobi
end

step = psize;
if psize == 1 
    psize = 0; % Gauss-Seidel
end

x = x0;
res = 1;
iter = 0;
diagelem = -diag(A);
norm_b = norm(b);

tic;
while (res > tol && iter < maxiter)
            iter = iter + 1;

            % Apply baby steps    
            for start = 1:step:n
                stop = start + psize;
                if (stop > n) 
                    stop = n;
                end
                
                    sum1 = diagelem(start:stop).*x(start:stop);     
                    sum2 = A(start:stop,1:n)*x;
                    sum = sum1+sum2;
                    x(start:stop) = (sum - b(start:stop))./ diagelem(start:stop) ;                                                                             
            end
                                                        
            res = norm(A*x-b)/norm_b;                      
end
elapsed = toc;

end





% *********** PJG Implementation for GPU **********
function [x, iter, res, elapsed]= PJG_GPU(A,b,maxiter,tol,x0,psize)
tic;

n = size(A,1);
if psize>n 
    psize = n; % Jacobi
end

step = psize;
if psize == 1 
    psize = 0; % Gauss-Seidel
end


x_gpu = gpuArray(x0);
b_gpu = gpuArray(b);
A_gpu = gpuArray(A);
diagelem_gpu = gpuArray(-diag(A));

res = 1;
iter = 0;

while (res > tol && iter < maxiter)
            iter = iter + 1;
            % Apply baby steps    
            for start = 1:step:n
                stop = start + psize;
                if (stop > n) 
                    stop = n;
                end                
                    sum1 = diagelem_gpu(start:stop).*x_gpu(start:stop);                         
                    sum2 = A_gpu(start:stop,1:n)*x_gpu;
                    sum = sum1+sum2;
                    x_gpu(start:stop) = (sum - b_gpu(start:stop))./ diagelem_gpu(start:stop) ;                                    
            end            
            res = gather(norm(A_gpu*x_gpu-b_gpu)/norm(b_gpu));              
end
x = gather(x_gpu);
elapsed = toc;    

end





function [x, iter, res, elapsed]= PJG_GPU_(A,b,maxiter,tol,x0,psize)
tic;

n = max(size(A));
if psize>n 
    psize = round(n/2);
end

step = psize;
if psize == 1 
    psize = 0;
end

x_gpu = gpuArray(x0);
b_gpu = gpuArray(b);
A_gpu = gpuArray(A');
diagelem_gpu = gpuArray(-diag(A));
norm_b = norm(b_gpu);

res = 1;
iter = 0;

while (res > tol && iter < maxiter)
            iter = iter + 1;
                        
            % Apply baby steps    
            for start = 1:step:n
                stop = start + psize;
                if (stop > n) 
                    stop = n;
                end
                
                sum1 = diagelem_gpu.*x_gpu;                
                sum2 = A_gpu(:,start:stop).'*x_gpu;
                
                sum = sum1(start:stop)+sum2;
                x_gpu(start:stop) = (sum - b(start:stop))./ diagelem_gpu(start:stop) ;                                                                                         
            end
            
            res = norm(A_gpu'*x_gpu-b_gpu)/norm_b;                      
end

x = gather(x_gpu);
res = gather(res);

elapsed = toc;

end