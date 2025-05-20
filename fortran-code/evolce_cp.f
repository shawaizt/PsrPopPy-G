	program thomas

	implicit none

	real*8 light, inertia, radius, pi, alpha_decay, secday, bk
	integer birthrate,update_step, seed_pid
	parameter (light=10.477,inertia=45.0,radius=6.0)
	parameter (pi=3.14159265, alpha_decay=1.0e7, update_step=1000)
	parameter (secday=86400.0*365.25, bk=19.505)
	parameter (birthrate=100)
	real*8 bfield, alpha, omega2, alphadot, K0, K, age, pipi4
	real*8 angle_part, p0, p1, brake, age_step_sec, brake_tt
	real*8 v0_new,v1_new,v0_old,v1_old,alpha0, brake0
	real*8 alpha_update,newbreak_part
	real*8 b, a_0, c_0, d_0, c_1, r10, light10, inertia10
	real ran1, normal, brake_mean, sigma, extra
	integer i,nup,c_nup,seed

c Generate a random seed
	call random_seed()
	call system_clock(count=seed)
	seed = -seed  ! Ensure seed is negative
	age_step_sec=birthrate*secday

c number of iterations per update
	nup = update_step / birthrate

c open the output file
	open(unit=10, file='evolve-{0}.out', status='replace')
        write(10, '(" ! Random seed: ", I0)') seed

c constant part
	K0 = log10(3.0) + 3.0*light + inertia - 6.0*radius
	pipi4 = 4.0*pi*pi
	alpha_update = exp(real(-birthrate)/alpha_decay)

c pick random alpha at birth
	  alpha0 = acos(ran1(seed))
	  alpha = alpha0
	  alphadot = -alpha0/alpha_decay/secday

c pick random period and period derivative
c hence nu and nudot and the magnetic field
	  p0 = normal(seed, 0.05, 0.01)
	  b = normal(seed, 12.7, 0.3)
	  b = 10.0**b
	  r10 = 10**radius
	  light10 = 10**light
	  inertia10 = 10**inertia
	  c_0 = 2 * pipi4 * r10**6
	  d_0 = 3 * light10**3 * inertia10 * p0
	  a_0 = c_0/d_0
	  p1 = a_0 * b**2 * sin(alpha0)**2
	  v0_old = 1./p0
	  v1_old = -p1/p0/p0
	  bfield = bk + log10(sqrt(p0*p1))
	  K = K0 - 2.0*bfield
	  K = 10**K

c pick random braking index at birth
	  brake0 = normal(seed,2.8,1.0)
	  brake = brake0

c age loop
	  age = 0.0
	  c_nup = 1
	  do i = 1,1000000

c update alpha and alphadot
	    alpha = alpha*alpha_update
	    alphadot = alphadot*alpha_update

c omega2 = rotational rate ^ 2
c braking index from the Tauris & Konar paper
	    omega2 = pipi4/p0/p0
	    angle_part = cos(alpha)/(sin(alpha)**3)
	    newbreak_part = K*alphadot*angle_part/omega2

c update braking index only if required
	    if (c_nup .eq. nup) then
	      brake_tt = brake0 - newbreak_part
c random perturbation on the braking index
	      sigma = brake_tt / 3.0
	      brake_mean = brake_tt
	      extra = normal(seed,brake_mean,sigma)
	      brake = extra
c	      brake = brake_tt
	      write(10,*)age/1.e6,p0,p1,alpha*180./pi,brake
              c_nup=1
	    else
	      c_nup = c_nup + 1
	      brake_tt = brake - newbreak_part
	      brake = brake_tt
	    endif

c update the period and period derivative
	    p0 = p0 + p1*age_step_sec
	    v0_new = 1./p0
	    v1_new = -v0_new/((age_step_sec*(brake-1.0))-v0_old/v1_old)
	    p1 = -v1_new*p0*p0
	    v0_old=v0_new
	    v1_old=v1_new

c update age
	    age = age + birthrate

	  enddo

	close(10)
	end
