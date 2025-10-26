% Function to set algorithm name and solver function handle
function [alg_name, alg_handle] = set_solver(alg_flag)
    switch alg_flag
        case 0
            alg_name = 'GD';
            alg_handle = @solve_GD;
        case 1
            alg_name = 'RGD';
            alg_handle = @solve_RGD;
        case 2
            alg_name = 'SGD';
            alg_handle = @solve_SGD;
        case 3
            alg_name = 'SubGD';
            alg_handle = @solve_SubGD;
        case 4
            alg_name = 'PGD';
            alg_handle = @solve_PGD;
        case 5
            alg_name = 'AP';
            alg_handle = @solve_AP;
        otherwise
            alg_name = 'UnknownAlg';
            error('Unknown alg_flag value. Use 0=GD, 1=RGD, 2=SGD, 3=SubGD, 4=PGD, 5=AP.');
    end
end