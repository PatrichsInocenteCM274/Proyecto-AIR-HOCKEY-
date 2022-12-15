from gym.envs.registration import register
register( 
    id='SimpleAirHockey-v0', 
    entry_point='simple_air_hockey.envs:SimpleAirHockeyEnv' 
)
