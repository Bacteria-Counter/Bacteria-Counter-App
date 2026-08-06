//
//  BacteriaDetectionView.swift
//  bacteriaapp
//
//  Created by Regina Celine Adiwinata on 06/08/26.
//

import SwiftUI

struct DetectionCountView: View {
    let count: Int

    var body: some View {
        SidebarSection(title: "DETECTION RESULT") {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Bacteria Detected")
                        .font(AppTheme.monoSmall)
                        .foregroundStyle(.secondary)
                    Text("\(count)")
                        .font(.system(size: 28, weight: .bold, design: .monospaced))
                        .foregroundStyle(AppTheme.accentGreen)
                }
                Spacer()
                Image(systemName: "circle.grid.3x3.fill")
                    .font(.system(size: 22))
                    .foregroundStyle(AppTheme.accentGreen.opacity(0.6))
            }
        }
    }
}
