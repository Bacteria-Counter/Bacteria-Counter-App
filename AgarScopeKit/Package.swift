// swift-tools-version:5.9
import PackageDescription

// Two targets rather than one, so the macOS app can link the engine instead of
// talking to a Python server over HTTP. The CLI stays because every number in
// README.md was produced through it -- the verification harness drives the same
// code the app runs, which is the only reason those figures say anything about
// the app.
let package = Package(
    name: "AgarScopeKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AgarScopeKit", targets: ["AgarScopeKit"]),
        .executable(name: "agarscope", targets: ["agarscope"]),
    ],
    targets: [
        .target(name: "AgarScopeKit", path: "Sources/AgarScopeKit"),
        .executableTarget(name: "agarscope", dependencies: ["AgarScopeKit"],
                          path: "Sources/agarscope"),
    ]
)
