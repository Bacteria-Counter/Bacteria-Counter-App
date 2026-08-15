import AgarScopeKit
import Foundation

// The whole CLI lives in the library so the verification flags exercise the
// same code the app links. See AgarScopeCLI.
exit(AgarScopeCLI.main(CommandLine.arguments))
