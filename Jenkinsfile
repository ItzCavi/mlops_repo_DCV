// Pipeline de integración continua del proyecto (Jenkins en Docker, ver docker-compose.yml).
//
// Etapas: clonar repositorio -> preparar Python -> validar estructura -> entorno virtual ->
//         estilo (ruff) -> entrenamiento -> pruebas con cobertura -> monitoreo de drift ->
//         SonarCloud -> imagen Docker de la API (solo en certification y master).
//
// Configuración recomendada del job: New Item > Multibranch Pipeline > Branch Sources: Git con la
// URL del repositorio (descubre developer, certification y master) y Script Path: Jenkinsfile.
// Alternativa: New Item > Pipeline > "Pipeline script from SCM" con una rama (*/developer, etc.).
//
// Trigger automático: Jenkins revisa el repositorio cada 5 minutos (pollSCM) y lanza el build
// cuando hay commits nuevos en la rama, por ejemplo al hacer merge de developer a certification
// o de certification a master. Se usa sondeo porque un Jenkins local no es accesible desde
// GitHub para recibir webhooks.
//
// SonarCloud (opcional): crear en Manage Jenkins > Credentials una credencial "Secret text" con
// ID "sonarcloud-token" y el token generado en sonarcloud.io. Si no existe, la etapa se omite y
// el build queda como UNSTABLE (amarillo), sin fallar.

def ramaActual() {
    // BRANCH_NAME existe en Multibranch Pipeline; GIT_BRANCH ("origin/master") en un job simple.
    return (env.BRANCH_NAME ?: env.GIT_BRANCH ?: '').replaceFirst('^origin/', '')
}

def dockerDisponible() {
    return sh(script: 'command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1', returnStatus: true) == 0
}

pipeline {
    agent any

    options {
        timeout(time: 40, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        skipDefaultCheckout(true)   // el clonado se hace en la etapa explícita "Clonar repositorio"
    }

    triggers {
        pollSCM('H/5 * * * *')
    }

    stages {
        stage('Clonar repositorio') {
            steps {
                deleteDir()
                script {
                    // En un job simple, checkout no publica GIT_BRANCH en env; se copia a mano.
                    def datosGit = checkout scm
                    env.GIT_BRANCH = env.GIT_BRANCH ?: datosGit.GIT_BRANCH
                    echo "Rama: ${ramaActual()} | commit: ${datosGit.GIT_COMMIT?.take(7)}"
                }
            }
        }

        stage('Preparar Python') {
            // La imagen jenkins/jenkins no trae venv; el contenedor corre como root (docker-compose),
            // así que puede instalarlo la primera vez.
            steps {
                sh '''
                    if ! python3 -m venv --help > /dev/null 2>&1; then
                        apt-get update
                        apt-get install -y --no-install-recommends python3 python3-venv python3-pip
                    fi
                    python3 --version
                '''
            }
        }

        stage('Validar estructura') {
            steps {
                script {
                    // Archivos que exige la estructura del proyecto. Si falta alguno, el pipeline se detiene.
                    def estructura = [
                        'mlops_pipeline/Cargar_datos.ipynb',
                        'mlops_pipeline/comprension_eda.ipynb',
                        'mlops_pipeline/ft_engineering.py',
                        'mlops_pipeline/model_training_evaluation.py',
                        'mlops_pipeline/model_deploy.py',
                        'mlops_pipeline/model_monitoring.py',
                        'Base_de_datos.csv',
                        'requirements.txt',
                        '.gitignore',
                        'setup.bat',
                        'readme.md',
                    ]
                    def faltantes = estructura.findAll { !fileExists(it) }
                    if (faltantes) {
                        error("La estructura del repositorio no cumple. Faltan: ${faltantes.join(', ')}")
                    }
                    echo 'Estructura del repositorio correcta.'
                }
            }
        }

        stage('Entorno virtual') {
            steps {
                sh '''
                    python3 -m venv venv
                    venv/bin/pip install --upgrade pip
                    venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Estilo de código') {
            // Convenciones de estilo y errores comunes (pycodestyle, pyflakes, bugbear, pyupgrade).
            steps {
                sh 'venv/bin/ruff check mlops_pipeline --line-length 120 --select E,F,W,B,UP --exclude "*.ipynb"'
            }
        }

        stage('Entrenamiento') {
            // Genera mejor_modelo.joblib, que necesitan las pruebas de la API y el monitoreo.
            steps {
                sh 'venv/bin/python mlops_pipeline/model_training_evaluation.py'
            }
        }

        stage('Pruebas') {
            steps {
                sh 'venv/bin/python -m pytest mlops_pipeline -v --cov=mlops_pipeline --cov-report=xml --junitxml=resultados_pruebas.xml'
            }
            post {
                always { junit 'resultados_pruebas.xml' }
            }
        }

        stage('Monitoreo de drift') {
            steps {
                sh 'venv/bin/python mlops_pipeline/model_monitoring.py --job'
            }
        }

        stage('SonarCloud') {
            when { expression { return dockerDisponible() } }
            steps {
                script {
                    try {
                        withCredentials([string(credentialsId: 'sonarcloud-token', variable: 'SONAR_TOKEN')]) {
                            // El scanner corre en su propio contenedor y ve el workspace gracias a
                            // --volumes-from jenkins (mismo volumen y misma ruta que el contenedor de Jenkins).
                            sh '''
                                docker run --rm --volumes-from jenkins -u 0 \
                                    -e SONAR_TOKEN -e SONAR_USER_HOME=/tmp/sonar \
                                    -w "$WORKSPACE" sonarsource/sonar-scanner-cli \
                                    -Dsonar.projectBaseDir="$WORKSPACE"
                            '''
                        }
                    } catch (err) {
                        unstable("No se ejecutó el análisis de SonarCloud: ${err.getMessage()}")
                    }
                }
            }
        }

        stage('Imagen Docker') {
            // La imagen de la API solo se construye en las ramas que van a producción.
            when { expression { return ramaActual() in ['certification', 'master'] && dockerDisponible() } }
            steps {
                sh 'docker build -t riesgo-impago-api:${BUILD_NUMBER} -t riesgo-impago-api:latest .'
            }
        }
    }

    post {
        always {
            archiveArtifacts allowEmptyArchive: true, fingerprint: true, artifacts:
                'mlops_pipeline/mejor_modelo.joblib, mlops_pipeline/resumen_evaluacion.csv, ' +
                'mlops_pipeline/comparacion_modelos.png, mlops_pipeline/monitoreo_*.csv, coverage.xml'
        }
    }
}
