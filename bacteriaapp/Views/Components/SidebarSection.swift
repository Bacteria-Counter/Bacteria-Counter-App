import SwiftUI

struct SidebarSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textMuted)
                .tracking(1.2)

            content
        }
    }
}

#Preview {
    SidebarSection(title: "DEVICE") {
        Text("Content")
    }
    .padding()
    .background(AppTheme.sidebarBackground)
}
