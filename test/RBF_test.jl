using SLFA

#Tests 1-5
@testset  "Evaluating a 1D Gaussian RBF" begin
    x0 = 1.0;
    w = 2.0;

    rbf_iso = Gaussian(x0, w)
    @test rbf_iso isa Gaussian{Isotropic, typeof(x0), 1}
    @test rbf_iso(x0) == 1.0
    @test rbf_iso(2) == 0.01831563888873418 #Add a rounding handler
    
    # Now we test it with parametric constructor
    rbf_iso_param = Gaussian{Isotropic,typeof(x0),1}(x0,w)
    @test rbf_iso isa Gaussian{Isotropic, typeof(x0), 1} 

    # Test with an array of parameters
    theta = [x0;w]
    rbf_iso_paramarray = Gaussian{Isotropic,typeof(x0),length(x0)}(theta)
    @test rbf_iso_paramarray isa Gaussian{Isotropic, typeof(x0), 1} 
    # eval_rbf_iso_paramarray_x0 = SLFA.eval_phi(x0,theta,rbf_iso_paramarray)
    # eval_rbf_iso_paramarray_2 = SLFA.eval_phi(2,theta,Gaussian{Isotropic,typeof(theta[1]),length(x0)})
    # @test eval_rbf_iso_paramarray_x0 == 1.0
    # @test eval_rbf_iso_paramarray_x == 0.01831563888873418
end

#Tests 6-9
@testset  "Evaluating a Vector/Scalar 1D Gaussian RBF" begin
    x0 = [1.0];
    w = 2.0;
    
    rbf_iso = Gaussian(x0, w)
    @test rbf_iso isa Gaussian{Isotropic, typeof(x0[1]), 1}
    @test rbf_iso(x0[1]) == 1.0 
    @test rbf_iso(2) == 0.01831563888873418 #Add a rounding handler here
    
    # Now we test it with parametric constructor
    rbf_iso_param = Gaussian{Isotropic,typeof(x0[1]),1}(x0,w)
    @test rbf_iso_param isa Gaussian{Isotropic, typeof(x0[1]), 1} 
end

#Tests 10-14
@testset  "Evaluating a ND Isotropic Gaussian RBF" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = 1.5;
    
    rbf_5D_iso = Gaussian(x0, w)
    @test rbf_5D_iso isa Gaussian{Isotropic, typeof(x0[1]), length(x0)}
    @test rbf_5D_iso(x0) == 1.0
    @test rbf_5D_iso([2.0,3.0,4.0,5.0,6.0]) == 1.300729765406762e-05
    
    # Test with parametric constructor
    rbf_iso_param = Gaussian{Isotropic,typeof(x0[1]),length(x0)}(x0,w)
    @test rbf_iso_param isa Gaussian{Isotropic, typeof(x0[1]), length(x0)}
    
    # Test with an array of parameters
    theta = [x0;w]
    rbf_iso_paramarray = Gaussian{Isotropic,typeof(x0[1]),length(x0)}(theta)
    @test rbf_iso_paramarray isa Gaussian{Isotropic, typeof(x0[1]), length(x0)} 
end

#Tests 15-17
@testset  "Evaluating a Vector/Vector 1D Gaussian RBF" begin
    x0 = [1.0];
    w = [2.0];
    rbf_iso = Gaussian(x0, w) #10,11,12
    @test rbf_iso isa Gaussian{Isotropic, typeof(x0[1]), 1} 
    @test rbf_iso(x0[1]) == 1.0 
    @test rbf_iso(2) == 0.01831563888873418 #Add a rounding handler here
end

#Tests 18-22
@testset "Evaluating an ND Anisotropic Gaussian RBF" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = [1.0, 1.5, 2.0, 2.5, 3.0];
    
    rbf_5D_aniso = Gaussian(x0, w)#13,14,15
    @test rbf_5D_aniso isa Gaussian{Anisotropic{Aligned}, typeof(x0[1]), length(x0)}
    @test rbf_5D_aniso(x0) == 1.0
    @test rbf_5D_aniso([2.0,3.0,4.0,5.0,6.0]) == 1.6918979226151304e-10
    
    # Test with parametric constructor
    rbf_5D_param_aniso = Gaussian{Anisotropic{Aligned},typeof(x0[1]),length(x0)}(x0,w)
    @test rbf_5D_param_aniso isa Gaussian{Anisotropic{Aligned},typeof(x0[1]),length(x0)}
    
    # Test with an array of parameters
    theta = [x0;w]
    rbf_aniso_paramarray = Gaussian{Anisotropic{Aligned},typeof(x0[1]),length(x0)}(theta)
    @test rbf_aniso_paramarray isa Gaussian{Anisotropic{Aligned}, typeof(x0[1]), length(x0)} 
end