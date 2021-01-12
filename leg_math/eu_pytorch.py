    pd.Series(losses).plot()
    pd.Series(test_losses).plot()

    pd.Series(accuracies).plot()
    pd.Series(test_accuracies).plot()

    true_ideal = random_votes[["coord1D", "coord2D"]]
    true_ideal.index = true_ideal.index.droplevel("vote_id")
    true_ideal = true_ideal.drop_duplicates()
    vote_data.keys()
    leg_crosswalk_rev = {v: k for k, v in vote_data["leg_crosswalk"].items()}
    true_ideal[["wnom1", "wnom2"]] = wnom_model.ideal_points[torch.tensor(true_ideal.index.map(leg_crosswalk_rev).values)].detach().numpy()

    true_ideal.corr()
    true_ideal.plot(kind='scatter', x="coord1D", y="wnom1")
    true_ideal.plot(kind='scatter', x="coord2D", y="wnom2")

    X = wnom_model.ideal_points[torch.arange(0, 100)].detach()
    Y = torch.tensor(true_ideal[["coord1D", "coord2D"]].values, dtype=torch.float)

    ab = torch.inverse(X.transpose(0, 1).mm(X))
    cd = X.transpose(0, 1).mm(Y)
    rot = ab.mm(cd)

    from scipy.linalg import orthogonal_procrustes
    rot, _ = orthogonal_procrustes(X, Y)
    temp_X = X.mm(torch.tensor(rot))

    true_ideal[["wnom1", "wnom2"]] = temp_X.numpy()
    true_ideal.corr()

    pd.DataFrame(temp_X.numpy()).plot(kind='scatter', x=0, y=1)

    (wnom_model.yes_points[torch.arange(0, 100)] ** 2).sum(dim=1).max()
    pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100)].detach().numpy()).plot(kind='scatter', x=0, y=1)
    wnom_model.w
    wnom_model.beta
    pd.Series(y_pred.detach().numpy()).hist()


    # For the time version
    wnom_model.ideal_points[torch.arange(0, 100)]
    time_tensor.unsqueeze(-1).shape
    wnom_model.ideal_points[legs] * time_tensor[votes]
    torch.sum(wnom_model.ideal_points[legs] * time_tensor[votes], axis=2)
    torch.sum(wnom_model.ideal_points[legs] * time_tensor[votes], axis=2).norm(dim=1)

    wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).norm(2, dim=1)

    pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).detach().numpy()).plot(kind="scatter", x=0, y=1)
    pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100), :, 0].detach().numpy()).plot(kind="scatter", x=0, y=1)
