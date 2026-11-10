"""Original frozen focal loss and evaluation functions, extracted without executing search code."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def focal_terms(pred,y,gamma,positive):
    from scipy.special import expit
    t=2.*y-1.;q=np.clip(expit(t*pred),1e-12,1.-1e-12);u=1.-q
    beta=np.where(y==1,positive,1.)
    value=-beta*u**gamma*np.log(q)
    grad=beta*t*u**gamma*(gamma*q*np.log(q)-u)
    hess=beta*u**gamma*q*(u*(2.*gamma+1.)+gamma*np.log(q)*(u-gamma*q))
    return value,grad,hess

def focal_objective(spec):
    stats=dict(calls=0,entries=0,hessian_floored=0,min_analytic_hessian=0.)
    def objective(pred,ds):
        _,grad,hess=focal_terms(pred,ds.get_label(),spec['gamma'],spec['positive_weight'])
        floor=spec['hessian_floor'];stats['calls']+=1;stats['entries']+=len(hess)
        stats['hessian_floored']+=int((hess<floor).sum())
        stats['min_analytic_hessian']=min(stats['min_analytic_hessian'],float(hess.min()))
        hess=np.maximum(hess,floor)
        weights=ds.get_weight()
        if weights is not None:grad=grad*weights;hess=hess*weights
        return grad,hess
    objective.stats=stats
    return objective

def threshold(y,p):
    f,t,c=roc_curve(y,p);ok=np.flatnonzero(np.isfinite(c))
    return float(c[ok[np.argmin(np.abs(1-f[ok]-t[ok]))]])

def metrics(y,p,cut):
    y=np.asarray(y,int);p=np.asarray(p,float)
    assert np.isfinite(p).all() and set(np.unique(y))=={0,1}
    pred=p>=cut;tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    result=dict(n=len(y),positives=int(y.sum()),AUC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),
                threshold=cut,Accuracy=(tn+tp)/len(y),Specificity=tn/(tn+fp),Sensitivity=tp/(tp+fn),F_score=2*tp/max(1,2*tp+fp+fn))
    order=np.argsort(-p,kind='stable');s=y[order]
    for fraction in (.05,.30):
        k=round(len(y)*fraction);tag=round(fraction*100);hit=int(s[:k].sum());discount=1/np.log2(np.arange(2,k+2))
        result.update({f'TP{tag}':hit,f'Precision{tag}':hit/k,f'Recall{tag}':hit/int(y.sum()),
                       f'NDCG{tag}':float(s[:k]@discount/discount[:min(k,int(y.sum()))].sum())})
    return result
