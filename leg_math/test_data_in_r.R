library(wnominate)
library(feather)
library(pscl)

# Install dwnominate package

# n_leg = 50
# n_votes = 500
# 
# test_data = generateTestData(legislators=n_leg,
#                              rcVotes=n_votes,
#                              # yea=matrix(runif(n_votes,min=-0.9,max=0.9),nrow=n_votes),
#                              # nay=matrix(runif(rcVotes,min=-0.9,max=0.9),nrow=n_votes),
#                              # dimweight=c(0.5),
#                              normal=1,
#                              seed=42)
# legis_data = test_data$legis.data

# data(sen90)
# test_data = dropRollCall(sen90, dropList=list(lop=3))

# Get the base data and then generate parties that are different
# party_sign = matrix(1, nrow=n_leg)
# party_sign[test_data$legis.data["party.1"] == 100] = -1
# 
# set.seed(65)
# temp_ideal = matrix(runif(n_leg), nrow=n_leg) * party_sign
# 
# test_data = generateTestData(legislators=n_leg,
#                              rcVotes=n_votes,
#                              dimweight=c(0.5),
#                              ideal=temp_ideal,
#                              normal=1,
#                              seed=42)
# test_data$legis.data = legis_data

# write.csv(test_data$votes, file="~/data/leg_math/test_votes.csv")
# write.csv(test_data$legis.data, file="~/data/leg_math/test_legislators.csv")

roll_call_mat = read.csv("~/data/leg_math/test_votes.csv", header=TRUE, row.names=1)
legis.data = read.csv("~/data/leg_math/test_legislators.csv", header=TRUE, row.names=1)
vote.data = read.csv("~/data/leg_math/test_vote_metadata.csv", header=TRUE, row.names=1)

for (i in seq(1, 3)){
    colnames(legis.data)[colnames(legis.data)==paste(c("coord", i, "D"), collapse="")] = paste(c("true_coord", i, "D"), collapse="")
}

test_data = rollcall(as.matrix(roll_call_mat), yea=1, nay=6, notInLegis=9,
                     legis.names = rownames(legis.data),
                     legis.data = legis.data,
                     vote.names = colnames(roll_call_mat),
                     vote.data = vote.data,
                     )

wnom1 = wnominate(test_data, polarity=c(1), dims=1, lop=0.01, minvotes=2)
write.csv(wnom1$legislators, file="~/data/leg_math/wnom1D_results.csv")
rownames(wnom1$rollcalls) = colnames(test_data$votes)
write.csv(wnom1$rollcalls, file="~/data/leg_math/wnom1D_rollcalls.csv")

wnom2 = wnominate(test_data, polarity=c(1, 2), dims=2)
rownames(wnom2$rollcalls) = colnames(test_data$votes)
write.csv(wnom2$legislators, file="~/data/leg_math/wnom2D_results.csv")
write.csv(wnom2$rollcalls, file="~/data/leg_math/wnom2D_rollcalls.csv")

wnom3 = wnominate(test_data, polarity=c(1, 2, 3), dims=3)
rownames(wnom3$rollcalls) = colnames(test_data$votes)
write.csv(wnom3$legislators, file="~/data/leg_math/wnom3D_results.csv")
write.csv(wnom3$rollcalls, file="~/data/leg_math/wnom3D_rollcalls.csv")

wnom4 = wnominate(test_data, polarity=c(1, 2, 3, 4), dims=4)
rownames(wnom4$rollcalls) = colnames(test_data$votes)
write.csv(wnom4$legislators, file="~/data/leg_math/wnom4D_results.csv")
write.csv(wnom4$rollcalls, file="~/data/leg_math/wnom4D_rollcalls.csv")

wnom5 = wnominate(test_data, polarity=c(1, 2, 3, 4, 5), dims=5)
rownames(wnom5$rollcalls) = colnames(test_data$votes)
write.csv(wnom5$legislators, file="~/data/leg_math/wnom5D_results.csv")
write.csv(wnom5$rollcalls, file="~/data/leg_math/wnom5D_rollcalls.csv")



asdf = ideal(test_data, d=1, normalize=TRUE, impute=TRUE, store.item=TRUE)
zxcv = idealToMCMC(asdf)
