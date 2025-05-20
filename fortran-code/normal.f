       real function normal(seed, mu, sigma)
           
       real mu, sigma, rnd1, rnd2
       integer*4 seed
  
       rnd1 = ran1(seed)
       rnd2 = ran1(seed)
       if(rnd1.eq.0.0) rnd1 = ran1(seed)
       normal = ((sigma * ((- (2.0 * log(rnd1))) ** 0.5)) * cos((2.0 *
     +   3.14159265) * rnd2)) + mu

       end
