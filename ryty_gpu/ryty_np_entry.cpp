// Entry for shadGT's NP (PlayStation Network) libraries, built for Ryty as one library. It is
// installed as libSceNpAuth.prx, the first NP library in GT Sport's load order, so Ryty's loader
// finds every NP export here before Ryty's own NP libraries or the stubs. With shadnet disabled the
// services answer as an offline PS4 does.

#include "core/libraries/system/commondialog.h"
#include "core/libraries/np/np_auth.h"
#include "core/libraries/np/np_commerce/np_commerce.h"
#include "core/libraries/np/np_common.h"
#include "core/libraries/np/np_manager.h"
#include "core/libraries/np/np_matching2/np_matching2.h"
#include "core/libraries/np/np_partner.h"
#include "core/libraries/np/np_party.h"
#include "core/libraries/np/np_profile_dialog/np_profile_dialog.h"
#include "core/libraries/np/np_score/np_score.h"
#include "core/libraries/np/np_signaling/np_signaling.h"
#include "core/libraries/np/np_sns_facebook_dialog.h"
#include "core/libraries/np/np_trophy.h"
#include "core/libraries/np/np_tus/np_tus.h"
#include "core/libraries/np/np_utility/np_utility.h"
#include "core/libraries/np/np_web_api/np_web_api.h"
#include "core/libraries/np/np_web_api2/np_web_api2.h"
#include "core/loader/symbols_resolver.h"
#include "core/user_settings.h"
#include "ryty_gpu/ryty_library.h"

namespace {

// Ryty's user service (libSceUserService.prx) numbers its players from this ID by slot.
constexpr s32 RytyFirstUserId = 0x10000000;

// shadNet signs in the logged-in users of user/users.json, by their user IDs. This library has its
// own copy of the user list, which nothing else loads or logs in (shadGT does that from its window
// and controllers), and the game asks for players by Ryty's user IDs. So: load the list, give each
// player Ryty's ID for its slot, and log them in before NpHandler starts. The IDs are changed in
// memory only; this library never saves the list.
void LoginRytyUsers() {
    UserSettings.Load();
    for (s32 index = 1; index <= Libraries::UserService::ORBIS_USER_SERVICE_MAX_LOGIN_USERS; ++index) {
        User* user = UserManagement.GetUserByPlayerIndex(index);
        if (!user)
            continue;
        user->user_id = RytyFirstUserId + index - 1;
        UserManagement.LoginUser(user, index);
        LOG_INFO(Lib_NpManager, "Ryty: player {} '{}' is user {:#x}, shadNet {}", index,
                 user->user_name, user->user_id,
                 user->shadnet_enabled ? "as " + user->shadnet_npid : std::string("off"));
    }
}

} // namespace

namespace Ryty {
void InstallGameLog(); // ryty_game_log.cpp
}

void Ryty::RytyLibraryInitialize() {
    InitializeCommon("shadgt_np.log");
    InstallGameLog(); // GTS-009, with RYGT_GAME_LOG=1
    LoginRytyUsers(); // before NpManager::RegisterLib starts NpHandler
    Core::Loader::SymbolsResolver resolver;
    using namespace Libraries::Np;
    NpCommon::RegisterLib(&resolver);
    NpManager::RegisterLib(&resolver);
    NpAuth::RegisterLib(&resolver);
    NpCommerce::RegisterLib(&resolver);
    NpMatching2::RegisterLib(&resolver);
    NpPartner::RegisterLib(&resolver);
    NpParty::RegisterLib(&resolver);
    NpProfileDialog::RegisterLib(&resolver);
    NpScore::RegisterLib(&resolver);
    NpSignaling::RegisterLib(&resolver);
    NpSnsFacebookDialog::RegisterLib(&resolver);
    NpTrophy::RegisterLib(&resolver);
    NpTus::RegisterLib(&resolver);
    NpUtility::RegisterLib(&resolver);
    NpWebApi::RegisterLib(&resolver);
    NpWebApi2::RegisterLib(&resolver);
    BindExports();
}
