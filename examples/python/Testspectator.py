#!/usr/bin/env python

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function

from time import sleep
from vizdoom import *
from random import choice
import itertools as it
game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

# game.load_config("../../examples/config/basic.cfg")
# game.load_config("../../examples/config/deadly_corridor.cfg")
game.load_config("../../examples/config/deepdeathmatch.cfg")
# game.load_config("../../examples/config/deathmatch.cfg")
# game.load_config("../../examples/config/defend_the_center.cfg")
# game.load_config("../../examples/config/defend_the_line.cfg")
# game.load_config("../../examples/config/health_gathering.cfg")
# game.load_config("../../examples/config/my_way_home.cfg")
# game.load_config("../../examples/config/predict_position.cfg")
# game.load_config("../../examples/config/take_cover.cfg")
# game.load_config("../../examples/config/cig.cfg")
cig = False



if cig:

    game.set_doom_map("map01")  # Limited deathmatch.
    #game.set_doom_map("map02")  # Full deathmatch.

    # Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
    game.add_game_args("-host 1 -deathmatch +timelimit 2.0 "
                    "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")

    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name AI +colorset 0")   






# Enables freelook in engine
game.add_game_args("+freelook 1")

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Enables spectator mode, so you can play. Sounds strange but it is the agent who is supposed to watch not you.
game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR)
#game.set_mode(Mode.PLAYER)

game.init()

bots = 7
#H = game.get_game_variable(GameVariable.HEALTH)
episodes = 10
print("")
for i in range(episodes):
    print("Episode #" + str(i + 1))
    total_reward = 0
    game.new_episode()
    while not game.is_episode_finished():
        s = game.get_state()
        img = s.image_buffer
        misc = s.game_variables
        n = game.get_available_buttons_size()
        actions = [list(a) for a in it.product([0, 1], repeat=n)]
        #game.advance_action()
        game.make_action(choice(actions))
        a = game.get_last_action()

        H = game.get_game_variable(GameVariable.HEALTH)
        if H > 99:
            reward = 30
        elif  H > 80:
            reward = 20.0 
        elif H > 60:
            reward = 10.0 
        elif H > 40:
            reward = -10.0 
        elif H > 20:
            reward = -20.0
        elif H > 1:
            reward = -30.0
        else:
            reward = -100.0

        #game.set_death_penalty(100)

        #D=game.set_death_penalty(100)
        a = game.get_last_action()
        r = game.get_last_reward()
        #r=H**2/300.0 
        #r = game.get_last_reward()
        print("state #" + str(s.number))
        print("game variables: ", misc)
        print("action:", a)
        print("reward:", r)
        print("---------------------")
        print("Health:", H)
        print("MY REWARD:", reward)
        print("=====================""\n\n")
      
    print("episode finished!")
    #print("total reward:", game.get_total_reward())
    print("total reward:", total_reward)
    print("************************")
    sleep(2.0)

game.close()
