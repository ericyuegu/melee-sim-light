#include "runtime/scalar.h"
#include "runtime/savestate.h"
#include "ft/types.h"
#include "ftGameWatch/forward.h"
#include "it/types.h"
#include "it/items/itgamewatchfire.h"
#include <baselib/gobj.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MSL_CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #condition); \
        return -1; \
    } \
} while (0)

static unsigned msl_torch_count(const MslCoreMatch* match)
{
    unsigned count = 0;
    for (HSD_GObj* gobj = match->gobj.entities->items; gobj; gobj = gobj->next) {
        if (((Item*) gobj->user_data)->kind == It_Kind_GameWatch_Fire) ++count;
    }
    return count;
}

static int msl_step(MslCoreMatch* match, const MslCoreInput* input)
{
    size_t used = match->memory.used;
    size_t allocations = match->memory.allocation_count;
    MSL_CHECK(msl_core_match_step(match, input, match->random_seed,
                                  &(MslCoreStageEvents) { 0 }) == 0);
    MSL_CHECK(match->memory.sealed && match->memory.used == used &&
              match->memory.allocation_count == allocations);
    return 0;
}

static int msl_torch_regression(MslCoreGameData* data, unsigned players)
{
    MslCoreMatch match = { 0 }, copy = { 0 };
    MslCoreMatchConfig config = { 0 };
    MslCoreInput input = { 0 };
    config.stage_id = 28;
    config.frame_id = -123;
    config.initial_random_seed = config.frame_pre_random_seed = 1;
    config.match_damage_ratio = 1;
    config.num_players = players;
    config.stock_count = 99;
    for (unsigned p = 0; p < players; ++p) config.players[p].char_id = 24;
    if (players == 2) config.players[1].char_id = 22;
    MSL_CHECK(msl_core_match_init(&match, data, &config, &input) == 0);
    if (players == 2) {
        // Real controller input: continuous same-action transitions skip the
        // torch removal state. The unpatched build aborts on frame 279.
        for (unsigned frame = 0; frame < 28924; ++frame) {
            input.p[0].c_x = frame >= 150 && frame % 2 == 0 ? -80 : 0;
            MSL_CHECK(msl_step(&match, &input) == 0);
        }
        MSL_CHECK(msl_torch_count(&match) >= 680);
    } else {
        // Challenge the per-fighter bound without allowing fighters to
        // interrupt each other's attacks. Spawn through the source owner.
        for (unsigned p = 0; p < players; ++p) {
            Fighter* fp = match.fighters[p]->user_data;
            fp->motion_id = ftGw_MS_AttackS4;
            for (unsigned item = 0; item < 700; ++item) {
                MSL_CHECK(itGamewatchFire_Spawn(match.fighters[p], &fp->cur_pos,
                                                FtPart_LHandNb, 1) != NULL);
            }
        }
        MSL_CHECK(msl_torch_count(&match) == 700 * players);
    }
    printf("torch reserve players=%u live=%u arena=%zu records=%u\n",
           players, msl_torch_count(&match), match.memory.used, match.relocation_count);
    MSL_CHECK(msl_core_match_init(&copy, data, &config, &(MslCoreInput) { 0 }) == 0);
    size_t size = msl_core_match_save_size(&match), written;
    void* saved = malloc(size);
    MSL_CHECK(saved != NULL);
    MSL_CHECK(msl_core_match_save(&match, saved, size, &written) == 0);
    MSL_CHECK(msl_core_match_restore(&copy, saved, written) == 0);
    free(saved);
    MSL_CHECK(msl_torch_count(&copy) == msl_torch_count(&match));
    memset(&input, 0, sizeof(input));
    for (unsigned frame = 0; frame < 90; ++frame) {
        MSL_CHECK(msl_step(&match, &input) == 0);
        MSL_CHECK(msl_step(&copy, &input) == 0);
        MSL_CHECK(memcmp(&match.output, &copy.output, sizeof(match.output)) == 0);
        MSL_CHECK(match.random_seed == copy.random_seed);
        MSL_CHECK(msl_torch_count(&match) == msl_torch_count(&copy));
    }
    MSL_CHECK(msl_torch_count(&match) == 0);
    msl_core_match_destroy(&copy);
    msl_core_match_destroy(&match);
    return 0;
}

int main(int argc, char** argv)
{
    MslCoreGameData* data = calloc(1, sizeof(*data));
    if (argc != 2 || data == NULL || msl_core_game_data_init(data, argv[1])) return 1;
    int result = msl_torch_regression(data, 2) || msl_torch_regression(data, 4);
    msl_core_game_data_deinit(data);
    free(data);
    return result != 0;
}
