function [result] =stability_clustering(da,k)

n=size(da,1);
[sim]=pair_relation(da,k);
% 
%  [sim] = kernal_sim(da);

m=-2*(0.5.*log(0.5));

sim=(sim-min(min(sim)))./(max(max(sim))-min(min(sim)));
th=graythresh(sim);


p=sim;
SH=zeros(n,n);

lo1=p<th&p~=0;
p1=p(lo1);
p1=p1.*(0.5./th);
SH(lo1)=-(p1.*log(p1)+(1-p1).*(log(1-p1)));

lo=p>=th&p~=1;
p2=p(lo);
p2=1-((1-p2).*(0.5./(1-th)));
SH(lo)=-(p2.*log(p2)+(1-p2).*(log(1-p2)));
SH=m-SH;

he=sum(SH)./n;
he=(he-min(he))./(max(he)-min(he));
th=1*graythresh(he);
[bin] =he>=th;

locat=find(bin==1);

% [sim] = kernal_sim(da);
% 
% sim=-squareform(pdist(da,'cosine'));
sim=-squareform(pdist(da));

sim=(sim-min(min(sim)))./(max(max(sim))-min(min(sim)));

data=(1:1:n)';
A=1:1:n;
de=setdiff(A,locat);
liu=data(locat,:);

newsim=sim(locat,locat);
% newsim_dist=1-newsim;
% newsim_dist=newsim_dist-diag(diag(newsim_dist));
% newsim_dist = squareform(newsim_dist);
% Z=linkage(newsim_dist,'single');
% 
% % I=inconsistent(Z,floor(sqrt(length(locat))));
% I=inconsistent(Z,4);
% [~,find_k]=max(I(:,end));
% detial_k=length(locat)+1-find_k(end);
% 
% detial_k=k;
% 
% if detial_k<k
%     detial_k=k;  
% end
% result=cluster(Z,'maxclust',detial_k);

% result=kmeans(da(locat,:),k);
% result=SpectralCluster(da(locat,:),k);
 
[result] = ap(newsim);
% result=ap(-squareform(pdist(da(locat,:))));

detial_k=length(unique(result));
liu=[liu result];


[newliu,newliudata]=de_result3(locat,liu,de,sim,detial_k,data);
data(newliu,end+1)=newliudata(:,end);
result=data(:,end);
detial_k=length(unique(result));
if detial_k~=k
    simij=zeros(1,(detial_k-1)*(detial_k-2)/2+(detial_k-1));
    for i=1:detial_k
        locat_i=result==i;
        for j=i+1:detial_k
            locat_j=result==j;
            sim_ij=sim(locat_i,locat_j);           
            ij=sum(sum(sim_ij))/(sum(locat_i)*sum(locat_j)); 
            simij((detial_k-1)*(i-1)-((i-1)*(i-2)/2)+j-i)=ij;
        end
    end
    simij=simij./max(simij);
    dis=1-simij;
    Z = linkage(dis,'single');
    clu=cluster(Z,k);
    CC=zeros(n,1);
    for i=1:detial_k
        biaoji=clu(i);
        l= result==i;
        CC(l,:)=biaoji;
    end
    result=CC;
end

end


function [newliu,newliudata]=de_result3(locat,liu,de,sim,k,data)
newliudata=liu;
deliu=sim(de,locat);
n=length(de);
relocat=zeros(n,k);
for i=1:k
    l= newliudata(:,end)==i;
    liu=deliu(:,l);
    max_liu=max(max(liu));
    min_liu=min(min(liu));
    if max_liu==min_liu
        liul=liu;
        liul(:)=1;
    else
        liul=(liu-min(min(liu)))./(max(max(liu))-min(min(liu)));
    end
    binliul=step1(liul);
    binliul_value=liu.*binliul;
    relocati=sum(binliul_value,2);
    ni=sum(binliul,2)+1;
    relocati=relocati./ni;

%     relocati=sum(liu,2);
%     relocati=relocati./sum(l);
   
    relocat(:,i)=relocati;
end

if sum(sum(relocat))==0
    sumzeros=ones(size(relocat,1),1);
    result=sumzeros.*(k+1);
    k=k+1;
else
    maxrelocat=max(relocat,[],2);
    maxrelocat=maxrelocat./max(maxrelocat);
    binsumrelocat=step1(maxrelocat);
    relocat=relocat(binsumrelocat,:);
    sumzeros=binsumrelocat==1;
    [~,result]=max(relocat,[],2);
end

newliu=de(sumzeros);
newde=setdiff(de,newliu);
newdeliudata=data(newliu,:);
newliu=[locat newliu];
newdeliudata(:,end+1)=result;
newliudata=[newliudata;newdeliudata];

if ~isempty(newde)
   [newliu,newliudata]=de_result3(newliu,newliudata,newde,sim,k,data);
end

end

function [sim]=pair_relation(data,k)
n=size(data,1);
% nei=floor(2*sqrt(n));
nei=floor(n/(2*k));
% [nei,~] =NNN(data);

% [dataed,~] = knnsearch(data,data,'K',nei,'Distance','cosine');
[dataed,~] = knnsearch(data,data,'K',nei);
z=zeros(n,n);
for i=1:n
    z(i,dataed(i,:))=1;
end
sim=z*z';
sim=sim./nei;

end

function [sim] = kernal_sim(data)

dist_line=pdist(data);
N=length(dist_line);
percent=8;
position=round(N*percent/100);
sda=sort(dist_line);
dc=sda(position);
% dc=std(dist_line);
dist=squareform(dist_line);
sim=exp(-(dist/(dc)).^2);
% sim=sim+sim';
sim=sim./max(max(sim));
end

function [cl] = ap(sim)
% s=-squareform(pdist(data));
p=median(sim).*ones(1,size(sim,1));
[idx,~,~,~]=apcluster(sim,p);
cl = relabel(idx);
end




%APCLUSTER Affinity Propagation Clustering (Frey/Dueck, Science 2007)
% [idx,netsim,dpsim,expref]=APCLUSTER(s,p) clusters data, using a set 
% of real-valued pairwise data point similarities as input. Clusters 
% are each represented by a cluster center data point (the "exemplar"). 
% The method is iterative and searches for clusters so as to maximize 
% an objective function, called net similarity.
% 
% For N data points, there are potentially N^2-N pairwise similarities; 
% this can be input as an N-by-N matrix 's', where s(i,k) is the 
% similarity of point i to point k (s(i,k) needn’t equal s(k,i)).  In 
% fact, only a smaller number of relevant similarities are needed; if 
% only M similarity values are known (M < N^2-N) they can be input as 
% an M-by-3 matrix with each row being an (i,j,s(i,j)) triple.
% 
% APCLUSTER automatically determines the number of clusters based on 
% the input preference 'p', a real-valued N-vector. p(i) indicates the 
% preference that data point i be chosen as an exemplar. Often a good 
% choice is to set all preferences to median(s); the number of clusters 
% identified can be adjusted by changing this value accordingly. If 'p' 
% is a scalar, APCLUSTER assumes all preferences are that shared value.
% 
% The clustering solution is returned in idx. idx(j) is the index of 
% the exemplar for data point j; idx(j)==j indicates data point j 
% is itself an exemplar. The sum of the similarities of the data points to 
% their exemplars is returned as dpsim, the sum of the preferences of 
% the identified exemplars is returned in expref and the net similarity 
% objective function returned is their sum, i.e. netsim=dpsim+expref.
% 
% 	[ ... ]=apcluster(s,p,'NAME',VALUE,...) allows you to specify 
% 	  optional parameter name/value pairs as follows:
% 
%   'maxits'     maximum number of iterations (default: 1000)
%   'convits'    if the estimated exemplars stay fixed for convits 
%          iterations, APCLUSTER terminates early (default: 100)
%   'dampfact'   update equation damping level in [0.5, 1).  Higher 
%        values correspond to heavy damping, which may be needed 
%        if oscillations occur. (default: 0.9)
%   'plot'       (no value needed) Plots netsim after each iteration
%   'details'    (no value needed) Outputs iteration-by-iteration 
%      details (greater memory requirements)
%   'nonoise'    (no value needed) APCLUSTER adds a small amount of 
%      noise to 's' to prevent degenerate cases; this disables that.
% 
% Copyright (c) B.J. Frey & D. Dueck (2006). This software may be 
% freely used and distributed for non-commercial purposes.
%          (RUN APCLUSTER WITHOUT ARGUMENTS FOR DEMO CODE)
function [idx,netsim,dpsim,expref]=apcluster(s,p,varargin)
    if nargin==0, % display demo
        fprintf('Affinity Propagation (APCLUSTER) sample/demo code\n\n');
        fprintf('N=100; x=rand(N,2); % Create N, 2-D data points\n');
        fprintf('M=N*N-N; s=zeros(M,3); % Make ALL N^2-N similarities\n');
        fprintf('j=1;\n');
        fprintf('for i=1:N\n');
        fprintf('  for k=[1:i-1,i+1:N]\n');
        fprintf('    s(j,1)=i; s(j,2)=k; s(j,3)=-sum((x(i,:)-x(k,:)).^2);\n');
        fprintf('    j=j+1;\n');
        fprintf('  end;\n');
        fprintf('end;\n');
        fprintf('p=median(s(:,3)); % Set preference to median similarity\n');
        fprintf('[idx,netsim,dpsim,expref]=apcluster(s,p,''plot'');\n');
        fprintf('fprintf(''Number of clusters: %%d\\n'',length(unique(idx)));\n');
        fprintf('fprintf(''Fitness (net similarity): %%g\\n'',netsim);\n');
        fprintf('figure; % Make a figures showing the data and the clusters\n');
        fprintf('for i=unique(idx)''\n');
        fprintf('  ii=find(idx==i); h=plot(x(ii,1),x(ii,2),''o''); hold on;\n');
        fprintf('  col=rand(1,3); set(h,''Color'',col,''MarkerFaceColor'',col);\n');
        fprintf('  xi1=x(i,1)*ones(size(ii)); xi2=x(i,2)*ones(size(ii)); \n');
        fprintf('  line([x(ii,1),xi1]'',[x(ii,2),xi2]'',''Color'',col);\n');
        fprintf('end;\n');
        fprintf('axis equal tight;\n\n');
        return;
    end;
    start = clock;
    % Handle arguments to function
    if nargin<2 error('Too few input arguments');
    else
        maxits=1000; convits=100; lam=0.9; plt=0; details=0; nonoise=0;
        i=1;
        while i<=length(varargin)
            if strcmp(varargin{i},'plot')
                plt=1; i=i+1;
            elseif strcmp(varargin{i},'details')
                details=1; i=i+1;
            elseif strcmp(varargin{i},'sparse')
    % 			[idx,netsim,dpsim,expref]=apcluster_sparse(s,p,varargin{:});
                fprintf('''sparse'' argument no longer supported; see website for additional software\n\n');
                return;
            elseif strcmp(varargin{i},'nonoise')
                nonoise=1; i=i+1;
            elseif strcmp(varargin{i},'maxits')
                maxits=varargin{i+1};
                i=i+2;
                if maxits<=0 error('maxits must be a positive integer'); end;
            elseif strcmp(varargin{i},'convits')
                convits=varargin{i+1};
                i=i+2;
                if convits<=0 error('convits must be a positive integer'); end;
            elseif strcmp(varargin{i},'dampfact')
                lam=varargin{i+1};
                i=i+2;
                if (lam<0.5)||(lam>=1)
                    error('dampfact must be >= 0.5 and < 1');
                end;
            else i=i+1;
            end;
        end;
    end;
    if lam>0.9
        fprintf('\n*** Warning: Large damping factor in use. Turn on plotting\n');
        fprintf('    to monitor the net similarity. The algorithm will\n');
        fprintf('    change decisions slowly, so consider using a larger value\n');
        fprintf('    of convits.\n\n');
    end;

    % Check that standard arguments are consistent in size
    if length(size(s))~=2 error('s should be a 2D matrix');
    elseif length(size(p))>2 error('p should be a vector or a scalar');
    elseif size(s,2)==3
        tmp=max(max(s(:,1)),max(s(:,2)));
        if length(p)==1 N=tmp; else N=length(p); end;
        if tmp>N
            error('data point index exceeds number of data points');
        elseif min(min(s(:,1)),min(s(:,2)))<=0
            error('data point indices must be >= 1');
        end;
    elseif size(s,1)==size(s,2)
        N=size(s,1);
        if (length(p)~=N)&&(length(p)~=1)
            error('p should be scalar or a vector of size N');
        end;
    else error('s must have 3 columns or be square'); end;

    % Construct similarity matrix
    if N>3000
        fprintf('\n*** Warning: Large memory request. Consider activating\n');
        fprintf('    the sparse version of APCLUSTER.\n\n');
    end;
    if size(s,2)==3 && size(s,1)~=3,
        S=-Inf*ones(N,N,class(s)); 
        for j=1:size(s,1), S(s(j,1),s(j,2))=s(j,3); end;
    else S=s;
    end;

    if S==S', symmetric=true; else symmetric=false; end;
    realmin_=realmin(class(s)); realmax_=realmax(class(s));

    % In case user did not remove degeneracies from the input similarities,
    % avoid degenerate solutions by adding a small amount of noise to the
    % input similarities
    if ~nonoise
        rns=randn('state'); randn('state',0);
        S=S+(eps*S+realmin_*100).*rand(N,N);
        randn('state',rns);
    end;

    % Place preferences on the diagonal of S
    if length(p)==1 for i=1:N S(i,i)=p; end;
    else for i=1:N S(i,i)=p(i); end;
    end;

    % Numerical stability -- replace -INF with -realmax
    n=find(S<-realmax_); if ~isempty(n), warning('-INF similarities detected; changing to -REALMAX to ensure numerical stability'); S(n)=-realmax_; end; clear('n');
    if ~isempty(find(S>realmax_,1)), error('+INF similarities detected; change to a large positive value (but smaller than +REALMAX)'); end;


    % Allocate space for messages, etc
    dS=diag(S); A=zeros(N,N,class(s)); R=zeros(N,N,class(s)); t=1;
    if plt, netsim=zeros(1,maxits+1); end;
    if details
        idx=zeros(N,maxits+1);
        netsim=zeros(1,maxits+1); 
        dpsim=zeros(1,maxits+1); 
        expref=zeros(1,maxits+1); 
    end;

    % Execute parallel affinity propagation updates
    e=zeros(N,convits); dn=0; i=0;
    if symmetric, ST=S; else ST=S'; end; % saves memory if it's symmetric
    while ~dn
        i=i+1; 

        % Compute responsibilities
        A=A'; R=R';
        for ii=1:N,
            old = R(:,ii);
            AS = A(:,ii) + ST(:,ii); [Y,I]=max(AS); AS(I)=-Inf;
            [Y2,I2]=max(AS);
            R(:,ii)=ST(:,ii)-Y;
            R(I,ii)=ST(I,ii)-Y2;
            R(:,ii)=(1-lam)*R(:,ii)+lam*old; % Damping
            R(R(:,ii)>realmax_,ii)=realmax_;
        end;
        A=A'; R=R';

        % Compute availabilities
        for jj=1:N,
            old = A(:,jj);
            Rp = max(R(:,jj),0); Rp(jj)=R(jj,jj);
            A(:,jj) = sum(Rp)-Rp;
            dA = A(jj,jj); A(:,jj) = min(A(:,jj),0); A(jj,jj) = dA;
            A(:,jj) = (1-lam)*A(:,jj) + lam*old; % Damping
        end;

        % Check for convergence
        E=((diag(A)+diag(R))>0); e(:,mod(i-1,convits)+1)=E; K=sum(E);
        if i>=convits || i>=maxits,
            se=sum(e,2);
            unconverged=(sum((se==convits)+(se==0))~=N);
            if (~unconverged&&(K>0))||(i==maxits) dn=1; end;
        end;

        % Handle plotting and storage of details, if requested
        if plt||details
            if K==0
                tmpnetsim=nan; tmpdpsim=nan; tmpexpref=nan; tmpidx=nan;
            else
                I=find(E); notI=find(~E); [tmp c]=max(S(:,I),[],2); c(I)=1:K; tmpidx=I(c);
                tmpdpsim=sum(S(sub2ind([N N],notI,tmpidx(notI))));
                tmpexpref=sum(dS(I));
                tmpnetsim=tmpdpsim+tmpexpref;
            end;
        end;
        if details
            netsim(i)=tmpnetsim; dpsim(i)=tmpdpsim; expref(i)=tmpexpref;
            idx(:,i)=tmpidx;
        end;
        if plt,
            netsim(i)=tmpnetsim;
            figure(234);
            plot(((netsim(1:i)/10)*100)/10,'r-'); xlim([0 i]); % plot barely-finite stuff as infinite
            xlabel('# Iterations');
            ylabel('Fitness (net similarity) of quantized intermediate solution');
    %         drawnow; 
        end;
    end; % iterations
    I=find((diag(A)+diag(R))>0); K=length(I); % Identify exemplars
    if K>0
        [tmp c]=max(S(:,I),[],2); c(I)=1:K; % Identify clusters
        % Refine the final set of exemplars and clusters and return results
        for k=1:K ii=find(c==k); [y j]=max(sum(S(ii,ii),1)); I(k)=ii(j(1)); end; notI=reshape(setdiff(1:N,I),[],1);
        [tmp c]=max(S(:,I),[],2); c(I)=1:K; tmpidx=I(c);
        tmpdpsim=sum(S(sub2ind([N N],notI,tmpidx(notI))));
        tmpexpref=sum(dS(I));
        tmpnetsim=tmpdpsim+tmpexpref;
    else
        tmpidx=nan*ones(N,1); tmpnetsim=nan; tmpexpref=nan;
    end;
    if details
        netsim(i+1)=tmpnetsim; netsim=netsim(1:i+1);
        dpsim(i+1)=tmpdpsim; dpsim=dpsim(1:i+1);
        expref(i+1)=tmpexpref; expref=expref(1:i+1);
        idx(:,i+1)=tmpidx; idx=idx(:,1:i+1);
    else
        netsim=tmpnetsim; dpsim=tmpdpsim; expref=tmpexpref; idx=tmpidx;
    end;
    if plt||details
        fprintf('\nNumber of exemplars identified: %d  (for %d data points)\n',K,N);
        fprintf('Net similarity: %g\n',tmpnetsim);
        fprintf('  Similarities of data points to exemplars: %g\n',dpsim(end));
        fprintf('  Preferences of selected exemplars: %g\n',tmpexpref);
        fprintf('Number of iterations: %d\n\n',i);
        fprintf('Elapsed time: %g sec\n',etime(clock,start));
    end;
    if unconverged
        fprintf('\n*** Warning: Algorithm did not converge. Activate plotting\n');
        fprintf('    so that you can monitor the net similarity. Consider\n');
        fprintf('    increasing maxits and convits, and, if oscillations occur\n');
        fprintf('    also increasing dampfact.\n\n');
    end;
end

function [cl] = relabel(idx)
    n=size(idx,1);
    un=unique(idx);
    k=length(un);
    cl=zeros(n,1);
    for i=1:k
        re=un(i);
        locat=idx==re;
        cl(locat,:)=i;
    end
end

function [r,nn_matrix] =NNN(data)
%3N-Q: Natural Nearest Neighbor with Quality

r=2;
flag=1;
n=size(data,1);
% [dataed,~] = knnsearch(data,data,'K',n,'Distance','cosine');
    [dataed,~] = knnsearch(data,data,'K',n);
    
while flag==1

    n_nei=dataed(:,2:r);

    if length(unique(n_nei))==n
        flag=0;
    end
    r=r+1;
end

nn_matrix=zeros(n,n);
% for i=1:n
%     [lo,~] = find(dataed==i);
%     nn_matrix(i,lo)=1;
% end

for i=1:n
     nn_matrix(i,dataed(i,:))=1;
end
end


function [binvat] =step1(vat)
th=graythresh(vat);
binvat=vat>th;
end