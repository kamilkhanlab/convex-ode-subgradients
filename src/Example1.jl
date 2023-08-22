#This implementation is to compute subgradients of the GMB relaxations of an
#original ODE system modified from [Scott and Barton, JOGO, 2013, Example~1]

#package declaration 
using IntervalArithmetic, EAGO, EAGO.McCormick
using DifferentialEquations, JuMP, Ipopt
using Plots; plotly()

#Time horizon for the original ODE system
tSpan = (0.0,4.0)

#Tolerance for ODE solver
e1 = 1e-6

#number of state variables
N = 2

#number of uncertain parameters
M = 2

#Lower and upper bounds for parameters
pL = (-3.0, 0.21)
pU = (3.0, 0.5)

#The ODE relaxations will be drawn w.r.t. p1, at p2 = 0.4
stepSize = (pU[1]-pL[1])/50
pSpan = pL[1]:stepSize:pU[1]
p2 = 0.4

#initial value function for a component xi
function x_initial(p, i)
    if i == 1
        return 0 * p[1] + 1.0
    elseif i == 2
        return 0 * p[1] + 0.5
    end
end

#right-hand side function for a component fi
function original_RHS(x, p, t, i)
    if i == 1
        dx = -(2+sin(p[1]/3))*(x[1])^2 + p[2]*x[1]*x[2]
    elseif i == 2
        dx = sin(p[1]/3)*(x[1])^2 - p[2]*abs(x[1]*x[2])
    end
    return dx
end

#Vector RHS of the original ODE system
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
        p = [pSpan[i],p2]
        paraSolution = [paraSolution; compute_original(p)]
    end
    return paraSolution
end

#funciton that constructs the subgradient propagation system's RHS for the GMB relaxations
#x is [xL, xU, xcv, xcc, xcv_grad, xcc_grad]
#pcomp is [p, p_I]
function GMB_subgradients_RHS(dx, x, pcomp, t)
    #construct unflattened McCormick and interval objects for x
    x_IA = Array{Interval}(undef, N)
    x_MC = Array{MC}(undef, N)
    for i = 1:1:N
        x_IA[i] = Interval(x[i], x[i+N])
        Sveccv = @SVector zeros(M)
        Sveccc = @SVector zeros(M)
        for j = 1:1:M
            Sveccv = setindex(Sveccv, x[4*N+(i-1)*M+j],j)
            Sveccc = setindex(Sveccc, x[4*N+N*M+(i-1)*M+j],j)
        end
        x_MC[i] = MC{M,NS}(x[i+2*N], x[i+3*N], x_IA[i], Sveccv, Sveccc, false)
    end
    
    #compute RHS functions for (xL_i, xU_i) and (xcc_i, xcv_i) for each index i
    for i = 1:1:N
        #compute dxi_L
        #flatten xi_IA to x_L
        x_IA[i] = Interval(x[i], x[i])
        dx_IA = original_RHS(x_IA, pcomp[1:M], t, i)
        dx[i] = lo(dx_IA)

        #compute dxi_U
        #flatten xi_IA to x_U
        x_IA[i] = Interval(x[i+N], x[i+N])
        dx_IA = original_RHS(x_IA, pcomp[1:M], t, i)
        dx[i+N] = hi(dx_IA)

        #unflatten xi_IA
        x_IA[i] = Interval(x[i], x[i+N])

        #compute dxi_cv and dxi_cv_grad
        #flatten xi_MC to xi_cv and xi_sub to xi_cv_grad
        Sveccv = @SVector zeros(M)
        for j = 1:1:M
            Sveccv = setindex(Sveccv, x[4*N+(i-1)*M+j],j)
        end
        x_MC[i] = MC{M,NS}(x[i+2*N], x[i+2*N], x_IA[i], Sveccv, Sveccv, false)
        dx_MC = original_RHS(x_MC, pcomp[M+1:2*M], t, i)
        dx[i+2*N] = dx_MC.cv
        for j = 1:1:M
            dx[4*N+(i-1)*M+j] = dx_MC.cv_grad[j]
        end

        #compute dxi_cc and dxi_cc_grad
        #flatten xi_MC to xi_cc and xi_sub to xi_cc_grad
        Sveccc = @SVector zeros(M)
        for j = 1:1:M
            Sveccc = setindex(Sveccc, x[4*N+N*M+(i-1)*M+j],j)
        end
        x_MC[i] = MC{M,NS}(x[i+3*N], x[i+3*N], x_IA[i], Sveccc, Sveccc, false)
        dx_MC = original_RHS(x_MC, pcomp[M+1:2*M], t, i)
        dx[i+3*N] = dx_MC.cc
        for j = 1:1:M
            dx[4*N+N*M+(i-1)*M+j] = dx_MC.cc_grad[j]
        end

        #unlatten xi_MC
        Sveccv = @SVector zeros(M)
        Sveccc = @SVector zeros(M)
        for j = 1:1:M
            Sveccv = setindex(Sveccv, x[4*N+(i-1)*M+j],j)
            Sveccc = setindex(Sveccc, x[4*N+N*M+(i-1)*M+j],j)
        end
        x_MC[i] = MC{M,NS}(x[i+2*N], x[i+3*N], x_IA[i],Sveccv, Sveccc, false)

        #It is verified that the resulting ODE relaxations do not visit the discrete jump
        #=
        #discrete jump
        if x[i+2*N] <= x[i]
            display(1)
            dx[i+2*N] = max(dx[i+2*N], dx[i])
        end
        if x[i+3*N] >= x[i+N]
            display(1)
            dx[i+3*N] = min(dx[i+3*N], dx[i+N])
        end
        =#
    end
end

#function that solves the subgradient system at a specific p
function compute_GMB_subgradients(p)
    #construct a parameter array pcomp:=[p_I, p_MC]
    pcomp = []
    for i = 1:1:M
        pcomp = push!(pcomp, Interval(pL[i], pU[i]))
    end
    for i = 1:1:M
        Svec = @SVector zeros(M)
        Svec = setindex(Svec, 1.0, i)
        pcomp = push!(pcomp, MC{M,NS}(p[i], p[i], Interval(pL[i], pU[i]),Svec,Svec,false))
    end

    #find the initial value [x0L, x0U, x0cv, x0cc] based on p_IA, p_MC and store in a
    #vector x0_MCobj
    x0_MCobj = Array{Float64}(undef, 4*N+2*N*M)
    for i = 1:1:N
        xi = x_initial(pcomp[M+1:2*M], i)
        x0_MCobj[i] = lo(xi.Intv)
        x0_MCobj[i+N] = hi(xi.Intv)
        x0_MCobj[i+2*N] = xi.cv
        x0_MCobj[i+3*N] = xi.cc
        for j = 1:1:M
            x0_MCobj[4*N+(i-1)*M+j] = xi.cv_grad[j]
            x0_MCobj[4*N+N*M+(i-1)*M+j] = xi.cc_grad[j]
        end
    end
    #solve the subgradient ODE problem
    prob = ODEProblem(GMB_subgradients_RHS, x0_MCobj, tSpan, pcomp)
    sol = DifferentialEquations.solve(prob,BS3(),reltol = e1)
    return sol.u[end, :]
end

#function that computes the parametric solution of the subgradient system
function compute_para_sol_GMB_subgradients()
    paraSolution = Array{Float64}(undef, 0, 1)
    for i = 1:1:length(pSpan)
        p = [pSpan[i],p2]
        paraSolution = [paraSolution; compute_GMB_subgradients(p)]
    end
    return paraSolution
end

#function that returns ODE convex relaxations and subgradients at a specific p
function compute_GMBcv_subgradients(p,i)
    xrelax = compute_GMB_subgradients(p)
    xcv = xrelax[1][i+2*N]
    xcv_sub = xrelax[1][4*N+(i-1)*M+1:4*N+(i-1)*M+2]
    return xcv, xcv_sub
end

#function that evaluates a single point on a subtangent line of cv
function compute_point_subtangent_GMBcv(pref,p,xcv,xcv_sub,i)
    #xcv,xcv_sub = compute_GMBcv_subgradients(pref,i)
    return xcv + xcv_sub[1]*(p[1]-pref[1]) + xcv_sub[2]*(p[2]-pref[2])
end

#function that computes the whole subtangent line of cv
function compute_point_subtangent_GMBcv(pref,i)
    xcv,xcv_sub = compute_GMBcv_subgradients(pref,i)
    #subpSpan = pref[1]-5*stepSize:stepSize:pref[1]+5*stepSize
    sublinecv = Array{Float64}(undef,length(subpSpan))
    for j = 1:1:length(subpSpan)
        p = [subpSpan[j],p2]
        sublinecv[j] = compute_point_subtangent_GMBcv(pref,p,xcv,xcv_sub,i)
    end
    return sublinecv
end

#function that returns ODE concave relaxations and subgradients at a specific p
function compute_GMBcc_subgradients(p,i)
    xrelax = compute_GMB_subgradients(p)
    xcc = xrelax[1][i+3*N]
    xcc_sub = xrelax[1][4*N+N*M+(i-1)*M+1:4*N+N*M+(i-1)*M+2]
    return xcc, xcc_sub
end

#function that evaluates a single point on a subtangent line of cc
function compute_point_subtangent_GMBcc(pref,p,xcc,xcc_sub,i)
    #xcv,xcv_sub = compute_GMBcv_subgradients(pref,i)
    return xcc + xcc_sub[1]*(p[1]-pref[1]) + xcc_sub[2]*(p[2]-pref[2])
end

#function that computes the whole subtangent line of cc
function compute_whole_subtangent_GMBcc(pref,i)
    xcc,xcc_sub = compute_GMBcc_subgradients(pref,i)
    #subpSpan = pref[1]-5*stepSize:stepSize:pref[1]+5*stepSize
    sublinecc = Array{Float64}(undef,length(subpSpan))
    for j = 1:1:length(subpSpan)
        p = [subpSpan[j],p2]
        sublinecc[j] = compute_point_subtangent_GMBcc(pref,p,xcc,xcc_sub,i)
    end
    return sublinecc
end


#Command for computing

#compute the original parametric solution
solori = compute_para_sol_original()
mmtrixori = hcat(solori...)
#compute parametric ODE relaxations
solu = compute_para_sol_GMB_subgradients()
mmtrixSB = hcat(solu...)

#Plot

#Plot for x1
index = 1
#Plot original parametric solution and ODE relaxations
 p1 = plot(pSpan, [mmtrixSB[index+2*N, :],mmtrixSB[index+3*N, :]],xlabel = "p<sub>1</sub>",ylabel = "x<sub>1</sub>",gridlinewidth = 2,guidefontsize = 15, xtickfontsize=15, ytickfontsize=15,line = (:red, :solid),linewidth = 2.5,framestyle=:box,label = false)
plot!(pSpan, mmtrixori[index,:], linewidth = 2.5,line = (:black, :dash), label = false)
#Compute subtangent line at reference points and plot
pref = [1.32,p2]
subpSpan = pref[1]-5*stepSize:stepSize:pref[1]+5*stepSize
sublinecv = compute_point_subtangent_GMBcv(pref,index)
sublinecc = compute_whole_subtangent_GMBcc(pref,index)
xcv,xcv_sub=compute_GMBcv_subgradients(pref,index)
xcc,xcc_sub=compute_GMBcc_subgradients(pref,index)
plot!([pref[1]], [xcv], seriestype = :scatter, color = :blue, label = false)
plot!([pref[1]], [xcc], seriestype = :scatter, color = :blue, label = false)
plot!(subpSpan, sublinecv, linewidth = 2.5,line = (:blue, :dot),label = false)
plot!(subpSpan, sublinecc, linewidth = 2.5,line = (:blue, :dot),label = false)
pref = [-0.84,p2]
subpSpan = pref[1]-5*stepSize:stepSize:pref[1]+5*stepSize
sublinecc = compute_whole_subtangent_GMBcc(pref,index)
sublinecv = compute_point_subtangent_GMBcv(pref,index)
xcv,xcv_sub=compute_GMBcv_subgradients(pref,index)
xcc,xcc_sub=compute_GMBcc_subgradients(pref,index)
plot!([pref[1]], [xcv], seriestype = :scatter, color = :blue, label = false)
plot!([pref[1]], [xcc], seriestype = :scatter, color = :blue, label = false)
plot!(subpSpan, sublinecv, linewidth = 2.5,line = (:blue, :dot),label = false)
plot!(subpSpan, sublinecc, linewidth = 2.5,line = (:blue, :dot),label = false)


#Plot for x2
index = 2
#Plot original parametric solution and ODE relaxations
p3 = plot(pSpan, mmtrixSB[index+2*N, :],xlabel = "p<sub>1</sub>",ylabel = "x<sub>2</sub>",gridlinewidth = 2,guidefontsize = 15, xtickfontsize=15, ytickfontsize=15,line = (:red, :solid),linewidth = 2.5,framestyle=:box,label = "GMB relaxations")
plot!(pSpan, mmtrixSB[index+3*N, :],line = (:red, :solid),linewidth = 2.5,label=false)
plot!(pSpan, mmtrixori[index,:], linewidth = 2.5,line = (:black, :dash), label = "Original model")
#Compute subtangent lines at reference points and plot
pref = [1.32,p2]
subpSpan = pref[1]-5*stepSize:stepSize:pref[1]+5*stepSize
sublinecv = compute_point_subtangent_GMBcv(pref,index)
sublinecc = compute_whole_subtangent_GMBcc(pref,index)
xcv,xcv_sub=compute_GMBcv_subgradients(pref,index)
xcc,xcc_sub=compute_GMBcc_subgradients(pref,index)
plot!([pref[1]], [xcv], seriestype = :scatter, color = :blue, label = false)
plot!([pref[1]], [xcc], seriestype = :scatter, color = :blue, label = false)
plot!(subpSpan, sublinecv, linewidth = 2.5,line = (:blue, :dot),label = false)
plot!(subpSpan, sublinecc, linewidth = 2.5,line = (:blue, :dot),label = false)
pref = [-0.72,p2]
subpSpan = pref[1]-5*stepSize:stepSize:pref[1]+5*stepSize
sublinecc = compute_whole_subtangent_GMBcc(pref,index)
sublinecv = compute_point_subtangent_GMBcv(pref,index)
xcv,xcv_sub=compute_GMBcv_subgradients(pref,index)
xcc,xcc_sub=compute_GMBcc_subgradients(pref,index)
plot!([pref[1]], [xcv], seriestype = :scatter, color = :blue, label = false)
plot!([pref[1]], [xcc], seriestype = :scatter, color = :blue, label = false)
plot!(subpSpan, sublinecv, linewidth = 2.5,line = (:blue, :dot),label = "Subtangents")
plot!(subpSpan, sublinecc, linewidth = 2.5,line = (:blue, :dot),label = false)


#Plot two subfigures
plot(p1, p3, layout = (1, 2))
plot!(legend=:outertopright, legendcolumns=3, tickfontsize=12,labelfontsize=12)

