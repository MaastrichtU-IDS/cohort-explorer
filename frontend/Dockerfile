FROM node:20-slim

ARG NEXT_PUBLIC_API_URL=http://localhost:3000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
ARG TARGETARCH
RUN case "${TARGETARCH}" in \
      arm64) swc_package="@next/swc-linux-arm64-gnu@14.2.3" ;; \
      amd64) swc_package="@next/swc-linux-x64-gnu@14.2.3" ;; \
      *) echo "Unsupported frontend build architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && npm install --no-save --package-lock=false "${swc_package}"

COPY . ./
RUN npm run build

EXPOSE 3000
CMD [ "npm", "run", "start" ]
