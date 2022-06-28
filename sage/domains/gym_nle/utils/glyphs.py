from enum import Enum
import pandas as pd
import numpy as np




#FOR SMALL ROOM:
#Nodes [isTerrain,isCreature,isItem,isDownstairs,isPlayer]
#So floor = [1,0,0,0,0], stairs = [1,0,0,+/-1,0], player = [0,1,0,0,1]
#Edges [Adjacent,Location,ContainedIn,Equipped,EW,NS]
# So Player -> location = [0,1,0,0,0,0]
# East -> [1,0,0,0,1,0], South -> [1,0,0,0,0,-1], etc 


#More generally
#Nodes: [isTerrain,isCreature,isItem]+[oneHotTerrain]+[oneHotCreature]+[oneHotItem]
#Edges [Adjacent,Location,ContainedIn,Equipped,EW,NS]

#so, we need to know for each of terrain, creature, and item:
#   1. How many options are there per type
#   2. How to map any given tile to it's encoding.

#Database: all glyphs, with type associated, description, and inclusion into different tilesets.
#then for a given tileset (part of the env description), we can query the DB, figure out what the right one-hot length and order is for each one. 
class GlyphTypes(Enum):
    Terrain = [1,0,0]
    Creature = [0,1,0]
    Item = [0,0,1]
    Blocking = [0,0,0]
#gt = Enum('GLYPH_TYPES','Terrain Creature Item Blocking')


GLYPH_DB = [
    [86, "mastodon", GlyphTypes.Creature,-1,0,1,0], #mastodon 
    # [283, "ghost", GlyphTypes.Creature,-1,0,0,1], #ghost 
    # [311, "djinn", GlyphTypes.Creature,-1,0,0,1], #djinn 

    [329, "player", GlyphTypes.Creature,-1,0,1,1],#player for lava
    [337, "player", GlyphTypes.Creature,-1,1,0,0],#player for small room

    [1923, "daggers", GlyphTypes.Item,73,0,0,1],
    [1935, "short sword", GlyphTypes.Item,73,0,0,1],
    [1965, "club", GlyphTypes.Item,73,0,1,1],
    [1975, "sling", GlyphTypes.Item,73,0,1,1],
    
    [2013, "chain mail", GlyphTypes.Item,72,0,0,1],
    [2019, "leather armor", GlyphTypes.Item,72,0,1,1],
    [2028, "robe", GlyphTypes.Item,72,0,0,1],

    #7 boots
    [2049, "combat boots", GlyphTypes.Item,72,0,0,1],
    [2050, "jungle boots", GlyphTypes.Item,72,0,0,1],
    [2051, "hiking boots", GlyphTypes.Item,72,0,0,1],
    [2052, "mud boots", GlyphTypes.Item,72,0,0,1],
    [2053, "buckled boots", GlyphTypes.Item,72,0,0,1],
    [2054, "riding boots", GlyphTypes.Item,72,0,0,1],
    [2055, "snow boots", GlyphTypes.Item,72,0,0,1],
    #28 rings
    [2056, "wooden ring", GlyphTypes.Item,51,0,0,1],
    [2057, "granite ring", GlyphTypes.Item,51,0,0,1],
    [2058, "opal ring", GlyphTypes.Item,51,0,0,1],
    [2059, "clay ring", GlyphTypes.Item,51,0,0,1],
    [2060, "coral ring", GlyphTypes.Item,51,0,0,1],
    [2061, "black onyx ring", GlyphTypes.Item,51,0,0,1],
    [2062, "moonstone ring", GlyphTypes.Item,51,0,0,1],
    [2063, "tiger eye ring", GlyphTypes.Item,51,0,0,1],
    [2064, "jade ring", GlyphTypes.Item,51,0,0,1],
    [2065, "bronze ring", GlyphTypes.Item,51,0,0,1],
    [2066, "agate ring", GlyphTypes.Item,51,0,0,1],
    [2067, "topaz ring", GlyphTypes.Item,51,0,0,1],
    [2068, "sapphire ring", GlyphTypes.Item,51,0,0,1],
    [2069, "ruby ring", GlyphTypes.Item,51,0,0,1],
    [2070, "diamond ring", GlyphTypes.Item,51,0,0,1],
    [2071, "pearl ring", GlyphTypes.Item,51,0,0,1],
    [2072, "iron ring", GlyphTypes.Item,51,0,0,1],
    [2073, "brass ring", GlyphTypes.Item,51,0,0,1],
    [2074, "copper ring", GlyphTypes.Item,51,0,0,1],
    [2075, "twisted ring", GlyphTypes.Item,51,0,0,1],
    [2076, "steel ring", GlyphTypes.Item,51,0,0,1],
    [2077, "silver ring", GlyphTypes.Item,51,0,0,1],
    [2078, "gold ring", GlyphTypes.Item,51,0,0,1],
    [2079, "ivory ring", GlyphTypes.Item,51,0,0,1],
    [2080, "emerald ring", GlyphTypes.Item,51,0,0,1],
    [2081, "wire ring", GlyphTypes.Item,51,0,0,1],
    [2082, "engagement ring", GlyphTypes.Item,51,0,0,1],
    [2083, "shiny ring", GlyphTypes.Item,51,0,0,1],

    
    [2098, "sack", GlyphTypes.Item,20,0,0,1],
    [2103, "lock pick", GlyphTypes.Item,20,0,0,1],

    [2131, "horn", GlyphTypes.Item,20,0,0,1],

    [2148, "meatball", GlyphTypes.Item,29,0,1,0],
    
    #25 potions
    [2178, "ruby potion", GlyphTypes.Item,52,0,0,1],
    [2179, "pink potion", GlyphTypes.Item,52,0,0,1],
    [2180, "orange potion", GlyphTypes.Item,52,0,0,1],
    [2181, "yellow potion", GlyphTypes.Item,52,0,0,1],
    [2182, "emerald potion", GlyphTypes.Item,52,0,0,1],
    [2183, "dark green potion", GlyphTypes.Item,52,0,0,1],
    [2184, "cyan potion", GlyphTypes.Item,52,0,0,1],
    [2185, "sky blue potion", GlyphTypes.Item,52,0,0,1],
    [2186, "brilliant blue potion", GlyphTypes.Item,52,0,0,1],
    [2187, "magenta potion", GlyphTypes.Item,52,0,0,1],
    [2188, "purple-red potion", GlyphTypes.Item,52,0,0,1],
    [2189, "puce potion", GlyphTypes.Item,52,0,0,1],
    [2190, "milky potion", GlyphTypes.Item,52,0,0,1],
    [2191, "swirly potion", GlyphTypes.Item,52,0,0,1],
    [2192, "bubbly potion", GlyphTypes.Item,52,0,0,1],
    [2193, "smoky potion", GlyphTypes.Item,52,0,0,1],
    [2194, "cloudy potion", GlyphTypes.Item,52,0,0,1],
    [2195, "effervescent potion", GlyphTypes.Item,52,0,0,1],
    [2196, "black potion", GlyphTypes.Item,52,0,0,1],
    [2197, "golden potion", GlyphTypes.Item,52,0,0,1],
    [2198, "brown potion", GlyphTypes.Item,52,0,0,1],
    [2199, "fizzy potion", GlyphTypes.Item,52,0,0,1],
    [2200, "dark potion", GlyphTypes.Item,52,0,0,1],
    [2201, "white potion", GlyphTypes.Item,52,0,0,1],
    [2202, "murky potion", GlyphTypes.Item,52,0,0,1],
    #27 wands
    [2289, "glass wand", GlyphTypes.Item,75,0,0,1],
    [2290, "balsa wand", GlyphTypes.Item,75,0,0,1],
    [2291, "crystal wand", GlyphTypes.Item,75,0,0,1],
    [2292, "maple wand", GlyphTypes.Item,75,0,0,1],
    [2293, "pine wand", GlyphTypes.Item,75,0,0,1],
    [2294, "oak wand", GlyphTypes.Item,75,0,0,1],
    [2295, "ebony wand", GlyphTypes.Item,75,0,0,1],
    [2296, "marble wand", GlyphTypes.Item,75,0,0,1],
    [2297, "tin wand", GlyphTypes.Item,75,0,0,1],
    [2298, "brass wand", GlyphTypes.Item,75,0,0,1],
    [2299, "copper wand", GlyphTypes.Item,75,0,0,1],
    [2300, "silver wand", GlyphTypes.Item,75,0,0,1],
    [2301, "platinum wand", GlyphTypes.Item,75,0,0,1],
    [2302, "iridium wand", GlyphTypes.Item,75,0,0,1],
    [2303, "zinc wand", GlyphTypes.Item,75,0,0,1],
    [2304, "aluminium wand", GlyphTypes.Item,75,0,0,1],
    [2305, "uranium wand", GlyphTypes.Item,75,0,0,1],
    [2306, "iron wand", GlyphTypes.Item,75,0,0,1],
    [2307, "steel wand", GlyphTypes.Item,75,0,0,1],
    [2308, "hexagonal wand", GlyphTypes.Item,75,0,0,1],
    [2309, "short wand", GlyphTypes.Item,75,0,0,1],
    [2310, "runed wand", GlyphTypes.Item,75,0,0,1],
    [2311, "long wand", GlyphTypes.Item,75,0,0,1],
    [2312, "curved wand", GlyphTypes.Item,75,0,0,1],
    [2313, "forked wand", GlyphTypes.Item,75,0,0,1],
    [2314, "spiked wand", GlyphTypes.Item,75,0,0,1],
    [2315, "jeweled wand", GlyphTypes.Item,75,0,0,1],

    [2351, "flint stones", GlyphTypes.Item,66,0,1,1],
    [2352, "rocks", GlyphTypes.Item,66,0,1,1],
    [2353, "boulder", GlyphTypes.Terrain,-1,0,1,1],


    [2359, "void", GlyphTypes.Blocking,-1,1,1,1],#blocking
    [2360, "horizontal wall", GlyphTypes.Blocking,-1,1,0,1],#blocking
    [2361, "vertical wall", GlyphTypes.Blocking,-1,1,0,1],#blocking
    [2362, "corner", GlyphTypes.Blocking,-1,1,0,1],#blocking
    [2363, "corner", GlyphTypes.Blocking,-1,1,0,1],#blocking
    [2364, "corner", GlyphTypes.Blocking,-1,1,0,1],#blocking
    [2365, "corner", GlyphTypes.Blocking,-1,1,0,1],#blocking
    [2376, "iron bars", GlyphTypes.Blocking,-1,0,1,0],#blocking

    [2378, "floor", GlyphTypes.Terrain,-1,1,1,1],#empty terrain
    [2379, "unlit floor", GlyphTypes.Terrain,-1,0,1,0],#empty terrain
    [2382, "stairs up", GlyphTypes.Terrain,-1,1,1,1], #terrain feature
    [2383, "stairs down", GlyphTypes.Terrain,-1,1,1,1], #terrain feature
    [2393, "lava", GlyphTypes.Terrain,-1,0,0,1], #terrain feature
    [2411, "pit trap", GlyphTypes.Terrain,-1,0,1,0],
    
]


#more:??
# 1965: a +1 club (weapon in hand)
# 1975: a +2 sling (alternate weapon; not wielded)
# 2351: 20 uncursed flint stones (in quiver pouch)
# 2352: 27 uncursed rocks
# 2019: a blessed +0 leather armor (being worn)

# 1935: a +0 short sword (weapon in hand)
# 1923: 12 +0 daggers (alternate weapon; not wielded)
# 2019: an uncursed +1 leather armor (being worn)
# 2194: an uncursed potion of sickness
# 2103: an uncursed lock pick
# 2098: an empty uncursed sack
class Glyphs:

    def __init__(self,tileset):
        self.tileset = tileset
        self.glyph_db = pd.DataFrame(GLYPH_DB,columns=['id','name','type','default_action','movement','minimal','lava']).set_index('id')
        self.BLOCKING_GLYPHS = (self.glyph_db[self.glyph_db['type']==GlyphTypes.Blocking].index).values
        self.TERRAIN_GLYPHS = (self.glyph_db[self.glyph_db['type']==GlyphTypes.Terrain].index).values
        self.KNOWN_GLYPHS = self.glyph_db.index.values
        self.onehot = pd.get_dummies((self.glyph_db[(self.glyph_db[tileset]==1) & (self.glyph_db['type']!=GlyphTypes.Blocking)]).index)

    def get_glyph_encoding(self,glyph):
        try:
            type_encoding = self.glyph_db['type'][glyph].value
            object_encoding = self.onehot[glyph]
        except KeyError:
            print(f"unknown glyph {glyph} encoding requested")
            type_encoding = self.glyph_db['type'][2019].value
            object_encoding = np.zeros_like(self.onehot[2378])

        return np.concatenate((type_encoding,object_encoding))

    def get_glyph_from_encoding(self,encoding):
        object_encoding = encoding[3:]
        row_id = (object_encoding==1).nonzero()[0].item()
        return self.onehot.columns[row_id]

    def get_default_action(self,glyph):
        try:
            return self.glyph_db['default_action'][glyph]
        except KeyError:
            print(f"unknown item {glyph} action requested")
            return 66


#inventory oclasses:
# RANDOM_CLASS =  0, /* used for generating random objects */
# ILLOBJ_CLASS =  1,
# WEAPON_CLASS =  2,
# ARMOR_CLASS  =  3,
# RING_CLASS   =  4,
# AMULET_CLASS =  5,
# TOOL_CLASS   =  6,
# FOOD_CLASS   =  7,
# POTION_CLASS =  8,
# SCROLL_CLASS =  9,
# SPBOOK_CLASS = 10, /* actually SPELL-book */
# WAND_CLASS   = 11,
# COIN_CLASS   = 12,
# GEM_CLASS    = 13,
# ROCK_CLASS   = 14,
# BALL_CLASS   = 15,
# CHAIN_CLASS  = 16,
# VENOM_CLASS  = 17,


# BLOCKING_GLYPHS = [
#     2359,
#     2360,
#     2361,
#     2362,
#     2363,
#     2364,
#     2365
# ]


#This is a dictionary of known glyphs.
# KNOWN_GLYPHS = {
#     329:"player",#player for lava
#     337:"player",#player for small room
#     2052:"",#lava item?
#     2131:"",#lava item?
#     2180:"",#lava item?
#     2197:"",#lava item?
#     2193:"",#lava item?
#     2289:"",#glass wand
#     2308:"",#lava item?
#     2311:"",#lava item?
#     2315:"",#lava item?
#     2359:"void", #blocking
#     2360:"horizontal wall", #blocking
#     2361:"vertical wall", #blocking
#     2362:"corner", #blocking
#     2363:"corner", #blocking
#     2364:"corner", #blocking
#     2365:"corner", #blocking
#     2378:"floor", #empty terrain
#     2382:"stairs up", #terrain feature
#     2383:"stairs down", #terrain feature
#     2393:"lava", #terrain feature
# }
