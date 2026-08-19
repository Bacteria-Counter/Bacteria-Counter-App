//
//  bacteriaappApp.swift
//  bacteriaapp
//

import SwiftUI

@main
struct bacteriaappApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1000, height: 680)
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unifiedCompact(showsTitle: false))
    }
}
