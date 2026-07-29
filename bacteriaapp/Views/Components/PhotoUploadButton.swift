import SwiftUI
import UniformTypeIdentifiers

struct PhotoUploadButton: View {
    let onPhotoSelected: (URL) -> Void
    let onError: (Error) -> Void

    @State private var isImporterPresented = false

    var body: some View {
        SecondaryButton(
            title: "Upload Photo",
            icon: "photo.badge.plus"
        ) {
            isImporterPresented = true
        }
        .fileImporter(
            isPresented: $isImporterPresented,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            do {
                guard let photoURL = try result.get().first else { return }
                onPhotoSelected(photoURL)
            } catch {
                onError(error)
            }
        }
    }
}

#Preview {
    PhotoUploadButton(
        onPhotoSelected: { _ in },
        onError: { _ in }
    )
    .padding()
    .frame(width: 220)
    .background(AppTheme.sidebarBackground)
}
