data(demo_asa24_simulated) # Note that this is a toy dataset. 


result <- simulated_annealing_combined(demo_asa24_simulated, candidate = 1, niter = 20, diet_score = "HEI2015") 
result$meal

result
