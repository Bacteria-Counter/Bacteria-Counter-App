//
//  ContentView.swift
//  bacteriaapp
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = MainViewModel()

    var body: some View {
        VStack(spacing: 0) {
            HeaderBarView(
                deviceName: viewModel.headerStatusText == "No Device" ? nil : viewModel.deviceName,
                isConnected: viewModel.isDeviceConnected
            )

            HStack(spacing: 0) {
                SidebarView(viewModel: viewModel)

                VStack(spacing: 0) {
                    MainViewportView(viewModel: viewModel)

                    StatusBarView(
                        message: viewModel.statusBarMessage,
                        icon: statusBarIcon
                    )
                }
            }
        }
        .background(AppTheme.background)
        .frame(minWidth: 900, minHeight: 600)
    }

    private var statusBarIcon: String {
        switch viewModel.appState {
        case .disconnected:
            "bolt.fill"
        case .connected:
            "waveform.path.ecg"
        case .analyzing:
            "waveform.path.ecg"
        case .complete:
            "checkmark.circle"
        }
    }
}

#Preview {
    ContentView()
}
