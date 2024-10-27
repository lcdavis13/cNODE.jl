#!/usr/bin/env julia

#SBATCH --nodes=25
#SBATCH --ntasks=274
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00

using Random

################################################################################
#       0. CONFIGURATION
################################################################################

# Setup initial configuration
begin
    num_workers = 4 # numbers larger than 4 don't seem to run simultaneously
    include("setup.jl")
end
# NOTE: first time running can take some minutes, packages installing

################################################################################
#       1. HYPERPARAMETER SEARCH
################################################################################

# Grid of values
@everywhere begin
    # Define potential hyperparameter values
    LearningRates = [[0.001,0.0025],[0.001,0.005],[0.001,0.01],[0.01,0.025],[0.01,0.05],[0.01,0.1]]
    LearningRates = [[0.01,0.025]] # overwrite with Ocean LRs from paper
    Minibatches = [1,5,10]
    Minibatches = [10] # overwrite with Ocean MB from paper
    # Parameters for run
    max_epochs = 500
    early_stoping = 500
    report = 50
    # Iterator over hyperparams, params and repetitions
    inx = collect(product(enumerate(LearningRates),enumerate(Minibatches)))[:]
    # Select "Drosophila_Gut" and "Soil_Vitro" as examples
    pars = real_data[6:6]
    # NOTE: variable `real_data` and values imported from module cNODE
    # "search hyperparameters..." |> println
end

# for (i,DATA) in enumerate(pars)
#     # Load percentage of dataset
#     path = "./data/real/$DATA/P.csv"
#     Z,P = import_data(path,2/3)

    # # Set a constant seed for reproducibility
    # Random.seed!(1234)
    # # Shuffle data (combine, shuffle, resplit)
    # data = [(Z[:, i], P[:, i]) for i in 1:size(Z, 2)]
    # shuffle!(data)
    # Z = hcat([d[1] for d in data]...)
    # P = hcat([d[2] for d in data]...)

#     N,M = size(Z)
#     # K = M
#     K = 5
#     # Explore hyperparameters in small dataset
#     "training $DATA..." |> println
#     for it in inx
#         ((j,lr),(k,mb)) = it

#         real_file = "real_sample_$(j)$(k).csv"
#         pred_file = "pred_sample_$(j)$(k).csv"
#         test_file = "test_loss_$(j)$(k).csv"
#         train_file = "train_loss_$(j)$(k).csv"
        
#         mb = Minibatches[k]
#         "LR: $lr MB: $mb" |> println
#         Qtst = SharedArray{Float64}(M,N)
#         Ptst = SharedArray{Float64}(M,N)
#         LossTrain = SharedArray{Float64}(K)
#         LossTest = SharedArray{Float64}(K)
#         # Use Leave-one-out cross validation
#         LeaveOneOut = kfolds((Z,P); k = K) |> enumerate |> collect
#         @sync @distributed for fold in LeaveOneOut
#             (l,((ztrn,ptrn),(ztst,ptst))) = fold
#             "training $l..."|>println
#             # Get cNODE
#             cnode = getModel(N)
#             # Train cNODE
#             W, loss_train, loss_val, loss_test = train_reptile(
#                                                     cnode, max_epochs,
#                                                     mb, lr,
#                                                     ztrn, ptrn, ztst, ptst, ztst, ptst,
#                                                     report, early_stoping
#                                                 )
            
#             # Save
#             LossTrain[l] = loss_train[end]
#             LossTest[l] = loss_test[end]
#             # println("LossTest ", LossTest[l])

#             mb_samples = size(ptst, 2)
#             for idx in 1:mb_samples
#                 Ptst[(l-1)*mb_samples + idx, :] = ptst[:, idx]'
#                 Qtst[(l-1)*mb_samples + idx, :] = predict(cnode, ztst[:, idx])'
#             end
#             # Report
#             # println(l,'\t',loss_train[end],'\t',loss_test[end])
#             # println('#' ^ 30)
#         end
#         # Save results
#         results = "./results/real/$DATA/hyperparameters/"
#         !ispath(results) && mkpath(results)
#         print("LossTest", LossTest)
#         writedlm(results*real_file, Ptst, ',')
#         writedlm(results*pred_file, Qtst, ',')
#         writedlm(results*test_file,  LossTest, ',')
#         writedlm(results*train_file, LossTrain, ',')
#     end
# end

################################################################################
#       2. Experimental Validation
################################################################################

for (i,DATA) in enumerate(pars)
    # Import full dataset
    path = "./data/real/$DATA/P.csv"
    Z,P = import_data(path)

    # Set a constant seed for reproducibility
    Random.seed!(1234)
    # Shuffle data (combine, shuffle, resplit)
    data = [(Z[:, i], P[:, i]) for i in 1:size(Z, 2)]
    shuffle!(data)
    Z = hcat([d[1] for d in data]...)
    P = hcat([d[2] for d in data]...)

    N,M = size(Z)
    K = M
    # K = 5
    # Select hyperparameters
    results = "./results/real/$DATA/hyperparameters/"
    # _mean = [ mean(readdlm(results*"test_loss_$i$j.csv",',',Float64,'\n')) for i in 1:6, j in 1:3] |> argmin
    # mb = Minibatches[_mean[2]]
    # lr = LearningRates[_mean[1]]
    mb = Minibatches[1]
    lr = LearningRates[1]
    # Allocate variables
    Qtst = SharedArray{Float64}(M,N)
    Ptst = SharedArray{Float64}(M,N)
    LossTrain = SharedArray{Float64}(K)
    LossTest = SharedArray{Float64}(K)
    # Run validation
    results = "./results/real/$DATA/"
    LeaveOneOut = kfolds((Z,P); k = K) |> enumerate |> collect
    "training $DATA..." |> println
    @sync @distributed for fold in LeaveOneOut
        (l,((ztrn,ptrn),(ztst,ptst))) = fold
        "training $l..."|>println
        # Get cNODE
        cnode = getModel(N)
        # Train cNODE
        W, loss_train, loss_val, loss_test = train_reptile(
                                                cnode, max_epochs,
                                                mb, lr,
                                                ztrn, ptrn, ztst, ptst, ztst, ptst,
                                                report, early_stoping
                                            )
        # Save values
        LossTrain[l] = loss_train[end]
        LossTest[l] = loss_test[end]
        
        mb_samples = size(ptst, 2)
        for idx in 1:mb_samples
            Ptst[(l-1)*mb_samples + idx, :] = ptst[:, idx]'
            Qtst[(l-1)*mb_samples + idx, :] = predict(cnode, ztst[:, idx])'
        end

        # Save realization
        !ispath(results*"loss_epochs/") && mkpath(results*"loss_epochs/")
        writedlm(results*"loss_epochs/train$l.csv",loss_train, ',')
        writedlm(results*"loss_epochs/test$l.csv",loss_test, ',')
        writedlm(results*"loss_epochs/val$l.csv",loss_val, ',')
        # Report
        println(i,'\t',loss_train[end],'\t',loss_test[end])
    end
    # Write full results
    writedlm(results*"real_sample.csv",Ptst, ',')
    writedlm(results*"pred_sample.csv",Qtst, ',')
    writedlm(results*"test_loss.csv",LossTest, ',')
    writedlm(results*"train_loss.csv",LossTrain, ',')
end
