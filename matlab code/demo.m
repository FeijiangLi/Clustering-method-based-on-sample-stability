load Iris.mat
gt=data(:,end);
k=length(unique(gt));
data_feature=data(:,1:end-1);
data_feature=predata(data_feature);
[cl] =stability_clustering(data_feature,k);
[ac1,ARI1,NMI1]=evaluate2(cl,gt,k)


load Flame.mat

gt=data(:,end);
k=length(unique(gt));
data_feature=data(:,1:end-1);
data_feature=predata(data_feature);
[cl] =stability_clustering_plot(data_feature,k);
[ac2,ARI2,NMI2]=evaluate2(cl,gt,k)