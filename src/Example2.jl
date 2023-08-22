#This implementation is to compute the subgradients of OB relaxations of an
#original ODE system 15.3.5 in Floudas et al. (1999)

#This implementation takes excessive long time to solve the subgradient ODE system at p=(14.0,8.5,2.0)
#for some unknown reason, but will eventually finish

#package declaration
using IntervalArithmetic, EAGO, EAGO.McCormick
using DifferentialEquations, JuMP, Ipopt
using Plots; plotly()

#Time horizon for the original ODE system
tSpan = (0.0,0.95)

#Tolerance for ODE solver and NLP solver
e1 = 1e-6
e2 = 1e-8

#number of state variables
N = 2

#number of uncertain parameters
M = 3

#Lower and upper bounds for parameters
pL = (11.0, 7.0, 1.0)
pU = (14.0, 10.0, 3.0)

#The ODE relaxations will be drawn w.r.t. p1, at median values of p2 and p3
stepSize = (pU[1]-pL[1])/10
pSpan = pL[1]:stepSize:pU[1]
p2 = (pL[2]+pU[2])/2
p3 = (pL[3]+pU[3])/2

#initial value function for a component i
function x_initial(p, i)
    if i == 1
        return 0 * p[1] + 1.0
    elseif i == 2
        return 0 * p[1] + 0.0
    end
end

#right-hand side function for a component fi
function original_RHS(x, p, t, i)
    if i == 1
        dx = -(p[1] + p[3]) * x[1]^2
    elseif i == 2
        dx = p[1] * x[1]^2 - p[2] * x[2]
    end
    return dx
end

#Vector RHS of the oroginal ODE system
function original_vector_RHS(dx,x,p,t)
    for i = 1:1:N
       dx[i] = original_RHS(x,p,t,i)
    end
end

#function that computes the original ODE solution at a specific p
function compute_original(p)
    x0Aug = [x_initial(p,1),x_initial(p,2)]
    prob = ODEProblem(original_vector_RHS, x0Aug, tSpan, p)
    sol = DifferentialEquations.solve(prob, BS3(),reltol = e1)
    return sol.u[end,:]
end

#function that computes the parametric solution at tf
function compute_para_sol_original()
    paraSolution = Array{Float64}(undef, 0, 1)
    for i = 1:1:length(pSpan)
        p = [pSpan[i],p2,p3]
        paraSolution = [paraSolution; compute_original(p)]
    end
    return paraSolution
end

#A function that allows using EAGO to construct McCormick objects in a certain way
function MC{N,T}(cv::Float64, cc::Float64, xI::Interval{Float64}) where {N, T <: RelaxTag}
    MC{N,T}(cv, cc, xI,
            zero(SVector{N,Float64}),
            zero(SVector{N,Float64}), false)
end

#function that constructs McCormick convex relaxations for the original RHS fi
#this construction already considers the equality constraint of the convex NLP employed at OB relaxation system's RHS by constructing a special zi_MC
#y is (z, xL, xU, xcv, p, pL, pU, tau(time point), i(index))
#y has a length of 4*N + M + 2
function fi_cv_flattened(y...)
    #correct the type of y
    if !isa(y[1], Array)
        y = collect(y)
    else
        y = y[1]
    end

    #construct MC object for z and p
    z_MC = Array{MC}(undef, N)
    p_MC = Array{MC}(undef, M)
    for j = 1:1:N
    z_MC[j] = MC{M+N,NS}(y[j],y[j],Interval(y[j+N],y[j+2*N]))
    end
    for j = 1:1:M
    p_MC[j] = MC{M+N,NS}(y[j+4*N],y[j+4*N],Interval(pL[j],pU[j]))
    end
    #special zi_MC
    i = floor(Int, y[end])
    z_MC[i] = MC{M+N, NS}(y[3*N+i],y[3*N+i],Interval(y[i+N],y[i+2*N]))
    df = original_RHS(z_MC, p_MC, y[end-1], i)
    return df.cv
end

#function that computes subgradients of fi_cv_flattened
function fi_cvGrad_flattened(grad, y...)
    if !isa(y[1], Array)
        y = collect(y)
    else
        y = y[1]
    end
    #construct MC object for z and p for computing subgradients
    z_MC = Array{MC}(undef, N)
    p_MC = Array{MC}(undef, M)
    for j = 1:1:N
        Svecz = @SVector zeros(N+M)
        Svecz = setindex(Svecz, 1.0, j)
    z_MC[j] = MC{M+N,NS}(y[j],y[j],Interval(y[j+N],y[j+2*N]),Svecz, Svecz, false)
    end
    for j = 1:1:M
        Svecp = @SVector zeros(N+M)
        Svecp = setindex(Svecp, 1.0, N+j)
    p_MC[j] = MC{M+N,NS}(y[j+4*N],y[j+4*N],Interval(pL[j],pU[j]),Svecp, Svecp, false)
    end

    #special zi_MC
    i = floor(Int, y[end])
    Sveczi = @SVector zeros(N+M)
    z_MC[i] = MC{M+N, NS}(y[3*N+i],y[3*N+i],Interval(y[i+N],y[i+2*N]),Sveczi, Sveczi, false)
    df = original_RHS(z_MC, p_MC, y[end-1], i)
    for i = 1:1:N
        grad[i] = df.cv_grad[i]
    end
end

#function that constructs McCormick concave relaxations for a component f_i
#this construction already considers the equality constraint of the convex NLP employed at OB relaxation system's RHS by constructing a special zi_MC
#y is (z, xL, xU, xcc, p, pL, pU, tau(time point), i(index))
#y has a length of 4*N + M + 2
function fi_cc_flattened(y...)
    #transfer tuple y into array y
    if !isa(y[1], Array)
        y = collect(y)
    else
        y = y[1]
    end
    # construct MC object for z and p
    z_MC = Array{MC}(undef, N)
    p_MC = Array{MC}(undef, M)
    for j = 1:1:N
    z_MC[j] = MC{M+N,NS}(y[j],y[j],Interval(y[j+N],y[j+2*N]))
    end
    for j = 1:1:M
    p_MC[j] = MC{M+N,NS}(y[j+4*N],y[j+4*N],Interval(pL[j],pU[j]))
    end
    # special zi_MC 
    i = floor(Int, y[end])
    z_MC[i] = MC{M+N, NS}(y[3*N+i],y[3*N+i],Interval(y[i+N],y[i+2*N]))
    df = original_RHS(z_MC, p_MC, y[end-1], i)
    return df.cc
end


#function that computes subgradients of fi_cc_flattened
function fi_ccGrad_flattened(grad, y...)
    if !isa(y[1], Array)
        y = collect(y)
    else
        y = y[1]
    end
    #construct MC object for z and p for computing subgradients
    z_MC = Array{MC}(undef, N)
    p_MC = Array{MC}(undef, M)
    for j = 1:1:N
        Svecz = @SVector zeros(N+M)
        Svecz = setindex(Svecz, 1.0, j)
    z_MC[j] = MC{M+N,NS}(y[j],y[j],Interval(y[j+N],y[j+2*N]),Svecz, Svecz, false)
    end
    for j = 1:1:M
        Svecp = @SVector zeros(N+M)
        Svecp = setindex(Svecp, 1.0, N+j)
    p_MC[j] = MC{M+N,NS}(y[j+4*N],y[j+4*N],Interval(pL[j],pU[j]),Svecp, Svecp, false)
    end
    #special zi_MC
    i = floor(Int, y[end])
    Sveczi = @SVector zeros(N+M)
    z_MC[i] = MC{M+N, NS}(y[3*N+i],y[3*N+i],Interval(y[i+N],y[i+2*N]),Sveczi, Sveczi, false)
    df = original_RHS(z_MC, p_MC, y[end-1], i)
    for i = 1:1:N
        grad[i] = df.cc_grad[i]
    end
end

#function that computes subgradients of of a unflattened fi_cv; that is without considering the equality constraint as in fi_cv_flattened.
function fi_cvGrad_unflattened(p,z,zL,zU,tau,i)
    #construct MC object for z and p for computing subgradients
    z_MC = Array{MC}(undef, N)
    p_MC = Array{MC}(undef, M)
    for j = 1:1:N
        Svecz = @SVector zeros(N+M)
        Svecz = setindex(Svecz, 1.0, j)
    z_MC[j] = MC{N+M,NS}(z[j],z[j],Interval(zL[j],zU[j]),Svecz, Svecz, false)
    end
    for j = 1:1:M
        Svecp = @SVector zeros(N+M)
        Svecp = setindex(Svecp, 1.0, N+j)
    p_MC[j] = MC{M+N,NS}(p[j],p[j],Interval(pL[j],pU[j]),Svecp, Svecp, false)
    end
    df = original_RHS(z_MC, p_MC, tau, i)
    grad = Array{Float64}(undef, N+M)
    for k = 1:1:N+M
        grad[k] = df.cv_grad[k]
    end
    return grad
end

#function that computes subgradients of of a unflattened fi_cc; that is without considering the equality constraint as in fi_cc_flattened.
function fi_ccGrad_unflattened(p,z,zL,zU,tau,i)
    #construct MC object for z and p for computing subgradients
    z_MC = Array{MC}(undef, N)
    p_MC = Array{MC}(undef, M)
    for j = 1:1:N
        Svecz = @SVector zeros(N+M)
        Svecz = setindex(Svecz, 1.0, j)
    z_MC[j] = MC{N+M,NS}(z[j],z[j],Interval(zL[j],zU[j]),Svecz, Svecz, false)
    end
    for j = 1:1:M
        Svecp = @SVector zeros(N+M)
        Svecp = setindex(Svecp, 1.0, N+j)
    p_MC[j] = MC{M+N,NS}(p[j],p[j],Interval(pL[j],pU[j]),Svecp, Svecp, false)
    end
    df = original_RHS(z_MC, p_MC, tau, i)
    grad = Array{Float64}(undef, N+M)
    for k = 1:1:N+M
        grad[k] = df.cc_grad[k]
    end
    return grad
end

#function that constructs subgradient ODE system RHS
#xAug is [xL, xU, xcv, xcc, xcv_grad, xcc_grad]
#pAug is [p, p_I]
function OB_subgradients_RHS(dxAug, xAug, pAug, t)
    #construct unflattened interval object
    xAug_I = Array{Interval}(undef, N)
    for i = 1:1:N
        xAug_I[i] = Interval(xAug[i], xAug[i+N])
    end
    #parametric NLP models for constructing optimization-based RHS
    model_cvi = Model(Ipopt.Optimizer)
    set_attribute(model_cvi, "print_level", 0)
    set_attribute(model_cvi, "tol", e2)
    set_attribute(model_cvi, "max_iter", 20)
    
    @NLparameter(model_cvi, xcv1[i=1:N] == 0.0)
    @variable(model_cvi, xAug[i+2*N]<=z1[i=1:N]<=xAug[i+3*N])
    @NLparameter(model_cvi, tau1 == 0.0)
    @NLparameter(model_cvi, zL1[i=1:N] == 0.0)
    @NLparameter(model_cvi, zU1[i=1:N] == 0.0)
    @NLparameter(model_cvi, pp1[i=1:M] == 0.0)
    @NLparameter(model_cvi, ii1 == 0)
    JuMP.register(model_cvi, :fi_cv, 4*N + M + 2, fi_cv_flattened, fi_cvGrad_flattened)
    @NLobjective(model_cvi, Min, fi_cv(z1...,zL1...,zU1...,xcv1...,pp1...,tau1,ii1))

    model_cci = Model(Ipopt.Optimizer)
    set_attribute(model_cci, "print_level", 0)
    set_attribute(model_cci, "tol", e2)
    set_attribute(model_cci, "max_iter", 20)
    @NLparameter(model_cci, xcc2[i=1:N] == 0.0)
    @variable(model_cci, xAug[i+2*N]<=z2[i=1:N]<=xAug[i+3*N])
    @NLparameter(model_cci, tau2 == 0.0)
    @NLparameter(model_cci, zL2[i=1:N] == 0.0)
    @NLparameter(model_cci, zU2[i=1:N] == 0.0)
    @NLparameter(model_cci, pp2[i=1:M] == 0.0)
    @NLparameter(model_cci, ii2 == 0)
    JuMP.register(model_cci, :fi_cc, 4*N + M + 2, fi_cc_flattened, fi_ccGrad_flattened)
    @NLobjective(model_cci, Max, fi_cc(z2...,zL2...,zU2...,xcc2...,pp2...,tau2,ii2))
    #compute dynamic for bounds and relaxations for each index i
    for i = 1:1:N
        #compute dxi_L
        #flatten xi_IA to x_L
        xAug_I[i] = Interval(xAug[i], xAug[i])
        dxAug_I = original_RHS(xAug_I, pAug[M+1:2*M], t, i)
        dxAug[i] = lo(dxAug_I)
        #compute dxi_U
        #flatten xi_IA to x_U
        xAug_I[i] = Interval(xAug[i+N], xAug[i+N])
        dxAug_I = original_RHS(xAug_I, pAug[M+1:2*M], t, i)
        dxAug[i+N] = hi(dxAug_I)
        #unflatten xi_IA
        xAug_I[i] = Interval(xAug[i], xAug[i+N])
        #compute dxicv
        JuMP.set_value(tau1, t)
        for j = 1:1:N
        JuMP.set_value(zL1[j], xAug[j])
        JuMP.set_value(zU1[j], xAug[j+N])
        JuMP.set_value(xcv1[j], xAug[j+2*N])
        end
        for j = 1:1:M
            JuMP.set_value(pp1[j], pAug[j])
        end
        JuMP.set_value(ii1, i)
        JuMP.optimize!(model_cvi)
        dxAug[i+2*N] = objective_value(model_cvi)
        #compute dsicv
        #get the optimal solution
        optim_cvi = value.(z1)
        optim_cvi[i] = xAug[i+2*N]
        #get the subgradient of f_icv at the optimal solution
        subgrad_fcvi = fi_cvGrad_unflattened(pAug[1:M],optim_cvi,xAug[1:N],xAug[N+1:2*N],t,i)
        #perform the subgradient theory of convex optimal-value functions
        sigmai_plus = Array{Float64}(undef, N)
        sigmai_minus = Array{Float64}(undef, N)
        for j = 1:1:N
            sigmai_plus[j] = max(subgrad_fcvi[j], 0.0)
            sigmai_minus[j] = min(subgrad_fcvi[j], 0.0)
        end
        sicv_deriv = zeros(Float64, M)
        sicv_deriv = sicv_deriv + subgrad_fcvi[N+1:end]
        for j = 1:1:N
            if j != i
            sicv_deriv = sicv_deriv + sigmai_plus[j]*xAug[4*N+(j-1)*M+1:4*N+j*M] + sigmai_minus[j]*xAug[4*N+N*M+(j-1)*M+1:4*N+N*M+j*M]
            else
                sicv_deriv = sicv_deriv + (sigmai_plus[j] + sigmai_minus[j])*xAug[4*N+(j-1)*M+1:4*N+j*M]
            end
        end
        #assign sicv_deriv to dxAug
            dxAug[4*N+(i-1)*M+1:4*N+i*M] = sicv_deriv[1:M]
        #compute dxicc
        JuMP.set_value(tau2, t)
        for j = 1:1:N
        JuMP.set_value(zL2[j], xAug[j])
        JuMP.set_value(zU2[j], xAug[j+N])
        #JuMP.set_value(xcv2[j], xAug[j+2*N])
        JuMP.set_value(xcc2[j], xAug[j+3*N])
        end
        for j = 1:1:M
            JuMP.set_value(pp2[j], pAug[j])
        end
        JuMP.set_value(ii2, i)
        JuMP.optimize!(model_cci)
        dxAug[i+3*N] = objective_value(model_cci)
        #compute dsicc
        #get the optimal solution
        optim_cci = value.(z2)
        optim_cci[i] = xAug[i+3*N]
        supergrad_fcci = fi_ccGrad_unflattened(pAug[1:M],optim_cci,xAug[1:N],xAug[N+1:2*N],t,i)
        #perform the subgradient theory of concave optimal-value functions
        sigmai_plus = Array{Float64}(undef, N)
        sigmai_minus = Array{Float64}(undef, N)
        for j = 1:1:N
            sigmai_plus[j] = max(supergrad_fcci[j], 0.0)
            sigmai_minus[j] = min(supergrad_fcci[j], 0.0)
        end
        sicc_deriv = zeros(Float64, M)
        sicc_deriv = sicc_deriv + supergrad_fcci[N+1:end]
        for j = 1:1:N
            if j != i
            sicc_deriv = sicc_deriv + sigmai_minus[j]*xAug[4*N+(j-1)*M+1:4*N+j*M] + sigmai_plus[j]*xAug[4*N+N*M+(j-1)*M+1:4*N+N*M+j*M]
            else
                sicc_deriv = sicc_deriv + (sigmai_plus[j] + sigmai_minus[j])*xAug[4*N+N*M+(j-1)*M+1:4*N+N*M+j*M]
            end
        end
        #assign sicc_deriv to dxAug
        dxAug[4*N+N*M+(i-1)*M+1:4*N+N*M+i*M] = sicc_deriv[1:M]
        #discrete jump is not visited
        #=
        discrete jump
        if xAug[i+2*N] <= xAug[i]
            dxAug[i+2*N] = max(dxAug[i+2*N], dxAug[i])
        end
        if xAug[i+3*N] >= xAug[i+N]
            dxAug[i+3*N] = min(dxAug[i+3*N], dxAug[i+N])
        end
        =#
    end
end

#function that computes subgradient system's solution at a fixed p
function compute_OB_subgradients(p)
    #construct a parameter array pAug:=[p,p_I]
    pAug = []
    for i = 1:1:M
        pAug = push!(pAug, p[i])
    end
    for i = 1:1:M
        pAug = push!(pAug, Interval(pL[i], pU[i]))
    end
    # find initial value x0_I, x0_MC based on p_I, p_MC and form x0Aug
    x0Aug = Array{Float64}(undef, 4*N+2*N*M)
    p_MC = Array{MC}(undef, M)
    for j = 1:1:M
        Svec = @SVector zeros(M)
        Svec = setindex(Svec, 1.0, j)
        p_MC[j] = MC{M,NS}(p[j],p[j],Interval(pL[j],pU[j]), Svec, Svec, false)
    end
    for i = 1:1:N
        xi = x_initial(p_MC,i)
        x0Aug[i] = lo(xi.Intv)
        x0Aug[i+N] = hi(xi.Intv)
        x0Aug[i+2*N] = xi.cv
        x0Aug[i+3*N] = xi.cc
        for j = 1:1:M
                x0Aug[4*N+(i-1)*M+j] = xi.cv_grad[j]
                x0Aug[4*N+N*M+(i-1)*M+j] = xi.cc_grad[j]
        end
    end
    prob = ODEProblem(OB_subgradients_RHS, x0Aug, tSpan, pAug)
    sol = DifferentialEquations.solve(prob, reltol = e1)
    return sol.u[end,:]
end

#function that computes parametric solutions of the subgradient system
function compute_para_sol_OB_subgradients()
    paraSolution = Array{Float64}(undef, 0, 1)
    for i = 1:1:length(pSpan)
        p = [pSpan[i],p2,p3]
        display(p)
        paraSolution = [paraSolution; compute_OB_subgradients(p)]
    end
    return paraSolution
end

#function that computes subgradient and OB relaxations at a fixed value of p
function compute_OBcv_subgradients(p,i)
    xrelax = compute_OB_subgradients(p)
    xcv = xrelax[1][i+2*N]
    xcv_sub = xrelax[1][4*N+(i-1)*M+1:4*N+(i-1)*M+3]
    return xcv, xcv_sub
end
#function that evaluates a single point on a subtangent line of OBcv
function compute_point_subtangent_OBcv(pref,p,xcv,xcv_sub,i)
    return xcv + xcv_sub[1]*(p[1]-pref[1]) + xcv_sub[2]*(p[2]-pref[2]) + xcv_sub[3]*(p[3]-pref[3])
end

#function that computes the whole subtangent line of OBcv
function compute_whole_subtangent_OBcv(pref,i)
    xcv,xcv_sub = compute_OBcv_subgradients(pref,i)
    sublinecv = Array{Float64}(undef,length(subpSpan))
    for j = 1:1:length(subpSpan)
        p = [subpSpan[j],p2,p3]
        sublinecv[j] = compute_point_subtangent_OBcv(pref,p,xcv,xcv_sub,i)
    end
    return sublinecv
end
function compute_OBcc_subgradients(p,i)
    xrelax = compute_OB_subgradients(p)
    xcc = xrelax[1][i+3*N]
    xcc_sub = xrelax[1][4*N+N*M+(i-1)*M+1:4*N+N*M+(i-1)*M+3]
    return xcc, xcc_sub
end
function compute_point_subtangent_OBcc(pref,p,xcc,xcc_sub,i)
    return xcc + xcc_sub[1]*(p[1]-pref[1]) + xcc_sub[2]*(p[2]-pref[2]) + xcc_sub[3]*(p[3]-pref[3])
end
function compute_whole_subtangent_OBcc(pref,i)
    xcc,xcc_sub = compute_OBcc_subgradients(pref,i)
    sublinecc = Array{Float64}(undef,length(subpSpan))
    for j = 1:1:length(subpSpan)
        p = [subpSpan[j],p2,p3]
        sublinecc[j] = compute_point_subtangent_OBcc(pref,p,xcc,xcc_sub,i)
    end
    return sublinecc
end
#compute original parametric solution
solori = compute_para_sol_original()
mmtrixori = hcat(solori...)
#compute parametric OB relaxations
solOB = compute_para_sol_OB_subgradients()
mmtrixOB = hcat(solOB...)

#Plot
#for x1
index = 1
#plot parametric original solution and OB relaxations
 p4 = plot(pSpan, [mmtrixOB[index+2*N, :],mmtrixOB[index+3*N, :]],xlabel = "p<sub>1</sub>",ylabel = "x<sub>1</sub>",gridlinewidth = 2,guidefontsize = 15, xtickfontsize=15, ytickfontsize=15,line = (:red, :solid),linewidth = 2.5,framestyle=:box,label = false)
plot!(pSpan, mmtrixori[index,:], linewidth = 2.5,line = (:black, :dash), label = false)

#compute subtangent lines at reference points and plot
pref = [12.8,p2,p3]
subpSpan = pref[1]-1*stepSize:stepSize:pref[1]+1*stepSize
sublinecv = compute_whole_subtangent_OBcv(pref,index)
sublinecc = compute_whole_subtangent_OBcc(pref,index)
xcv,xcv_sub=compute_OBcv_subgradients(pref,index)
xcc,xcc_sub=compute_OBcc_subgradients(pref,index)
plot!([pref[1]], [xcv], seriestype = :scatter, color = :blue, label = false)
plot!([pref[1]], [xcc], seriestype = :scatter, color = :blue, label = false)
plot!(subpSpan, sublinecv, linewidth = 2.5,line = (:blue, :dot),label = false)
plot!(subpSpan, sublinecc, linewidth = 2.5,line = (:blue, :dot),label = false)
pref = [11.9,p2,p3]
subpSpan = pref[1]-1*stepSize:stepSize:pref[1]+1*stepSize
sublinecv = compute_whole_subtangent_OBcv(pref,index)
sublinecc = compute_whole_subtangent_OBcc(pref,index)
xcv,xcv_sub=compute_OBcv_subgradients(pref,index)
xcc,xcc_sub=compute_OBcc_subgradients(pref,index)
plot!([pref[1]], [xcv], seriestype = :scatter, color = :blue, label = false)
plot!([pref[1]], [xcc], seriestype = :scatter, color = :blue, label = false)
plot!(subpSpan, sublinecv, linewidth = 2.5,line = (:blue, :dot),label = false)
plot!(subpSpan, sublinecc, linewidth = 2.5,line = (:blue, :dot),label = false)

#for x2
index = 2
#plot parametric original solution and OB relaxations
 p5 = plot(pSpan, mmtrixOB[index+2*N, :],xlabel = "p<sub>1</sub>",ylabel = "x<sub>2</sub>",gridlinewidth = 2,guidefontsize = 15, xtickfontsize=15, ytickfontsize=15,line = (:red, :solid),linewidth = 2.5,framestyle=:box,label = "OB relaxations")
plot!(pSpan, mmtrixOB[index+3*N, :],line = (:red, :solid),linewidth = 2.5,label=false)
 plot!(pSpan, mmtrixori[index,:], linewidth = 2.5,line = (:black, :dash), label = "Original model")
#compute subtangent lines at reference points and plot
pref = [12.8,p2,p3]
subpSpan = pref[1]-1*stepSize:stepSize:pref[1]+1*stepSize
sublinecv = compute_whole_subtangent_OBcv(pref,index)
sublinecc = compute_whole_subtangent_OBcc(pref,index)
xcv,xcv_sub=compute_OBcv_subgradients(pref,index)
xcc,xcc_sub=compute_OBcc_subgradients(pref,index)
plot!([pref[1]], [xcv], seriestype = :scatter, color = :blue, label = false)
plot!([pref[1]], [xcc], seriestype = :scatter, color = :blue, label = false)
plot!(subpSpan, sublinecv, linewidth = 2.5,line = (:blue, :dot),label = false)
plot!(subpSpan, sublinecc, linewidth = 2.5,line = (:blue, :dot),label = false)
pref = [11.9,p2,p3]
subpSpan = pref[1]-1*stepSize:stepSize:pref[1]+1*stepSize
sublinecv = compute_whole_subtangent_OBcv(pref,index)
sublinecc = compute_whole_subtangent_OBcc(pref,index)
xcv,xcv_sub=compute_OBcv_subgradients(pref,index)
xcc,xcc_sub=compute_OBcc_subgradients(pref,index)
plot!([pref[1]], [xcv], seriestype = :scatter, color = :blue, label = false)
plot!([pref[1]], [xcc], seriestype = :scatter, color = :blue, label = false)
plot!(subpSpan, sublinecv, linewidth = 2.5,line = (:blue, :dot),label = "Subtangents")
plot!(subpSpan, sublinecc, linewidth = 2.5,line = (:blue, :dot),label = false)

#plot two subfigures
plot(p4, p5, layout = (1, 2))
plot!(legend=:outertopright, legendcolumns=3, tickfontsize=12,labelfontsize=12)
